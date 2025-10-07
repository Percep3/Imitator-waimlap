from tqdm import tqdm
import typing as t

from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
import torch
import random

torch.manual_seed(23)
random.seed(23)

from scripts.settings import DEBUG
from torch.optim.swa_utils import AveragedModel
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F

from src.mslm.utils.early_stopping import EarlyStopping
from src.mslm.checkpoint.manager import CheckpointManager
# from src.mslm.training import imitator_loss
from src.mslm.training.clip_loss import ClipContrastiveLoss
import nvtx
from datetime import datetime

def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x:        [B, T, D]
    mask: [B, T] con True = PAD (ignorar)
    """
    valid = (~mask).unsqueeze(-1).to(x.dtype)   # 1.0 en válidos
    num = (x * valid).sum(dim=dim)
    den = valid.sum(dim=dim).clamp_min(1e-6)
    return num / den

class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate, save_tb_model=True, **kwargs):
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
            mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
            dynamic=True
        )

        #Accelerator module
        self.accelerator = Accelerator(mixed_precision="bf16", dynamo_plugin=dynamo_plugin)
        self.device = self.accelerator.device

        #Hyperparameters
        self.epochs = kwargs.get("epochs", 100)
        self.learning_rate = learning_rate

        #Loggers
        self.log_interval = kwargs.get("log_interval", 5)
        self.save_tb_model = save_tb_model

        version = kwargs.get("model_version", 1)
        checkpoint = kwargs.get("checkpoint", 1)

        self.writer = SummaryWriter(f"../outputs/reports/{version}/{checkpoint}/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
        self.graph_added = False
        
        #Save and checkpoint
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 5)
        self.ckpt_mgr = CheckpointManager(
            kwargs.get("model_dir", "../outputs/checkpoints"),
            version,
            checkpoint,
        )

        #Loss Function
        if kwargs.get("compile", True):
            self.criterion = torch.compile(
                ClipContrastiveLoss(),
                backend="inductor",
                mode="default",
                dynamic=True
            )            
        else:
            self.criterion = ClipContrastiveLoss()
        

        #Model
        self.model = model
        self.load_previous_model = kwargs.get("load_previous_model", False)

        #DataLoaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        #Stopper
        self.early_stopping = EarlyStopping(patience=100)

        #Optimizer
        self.optimizer = None
        self.scheduler = None

        #Batch Sampling
        self.batch_size = kwargs.get("batch_size", 5)
        self.batch_sampling = kwargs.get("batch_sampling", True)
        if self.batch_sampling:
            self.sub_batch = kwargs.get("batch_sample", 4)

        #Options 
        self.prof = False
        
        self.grad_clip = kwargs.get("grad_clip", 0.1)
        self.weight_decay = kwargs.get("weight_decay", 0.05)

    def prepare_trainer(self):
        """Prepara todo lo necesario para el entrenamiento."""
        (self.model,
        self.criterion,
        self.optimizer,
        self.scheduler,
        self.train_loader,
        self.val_loader) = self.accelerator.prepare(
            self.model, self.criterion, self.optimizer, self.scheduler,
            self.train_loader, self.val_loader
        )

        self.ema = AveragedModel(self.model, avg_fn=lambda ema, p, n: ema*0.999 + p*0.001)
        
    @nvtx.annotate("Training Section", color="green")
    def train(self, prof = False):
        """Entrena el modelo Imitator.
        returns:
            train_loss: float, loss de entrenamiento
            val_loss: float, loss de validación
        """
        print("LR:", self.learning_rate)
        self.optimizer = AdamW(
            [
                {"params": self.model.parameters(), "weight_decay": self.weight_decay},
                {"params": [self.criterion.logit_scale], "weight_decay": 0.0,},
            ],
            lr=self.learning_rate,
            foreach=True
        )
        
        def linear_warmup_cosine_decay(current_step, warmup_steps, total_steps):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 0.5 * (1.0 + torch.cos(
                torch.tensor((current_step - warmup_steps) / (total_steps - warmup_steps) * 3.1415926535))
            ).item()

        warmup_steps = 2 * len(self.train_loader)  # p.ej. 10 epochs de warm-up
        total_steps = self.epochs * len(self.train_loader)

        lr_lambda = lambda step: linear_warmup_cosine_decay(step, warmup_steps, total_steps)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        if self.load_previous_model: 
            self.ckpt_mgr.load_checkpoint(self.model, self.optimizer, self.scheduler)

        self.prepare_trainer()
        self.prof = prof

        val_loss = 0
        train_loss = 0
        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch)
            val_loss = self._val(epoch)

            if epoch == 1:
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)
            elif epoch == self.epochs - 1:
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)
            elif (epoch % self.checkpoint_interval == 0 and epoch != 0) :
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)
            elif self.early_stopping.stop:
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)

            if self.early_stopping.stop:
                break

        return train_loss, val_loss

    @nvtx.annotate("Train: Train Epoch", color="green")
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        accumulated_metrics = {
            'acc_v2t': 0.0,
            'acc_t2v': 0.0,
            'temp': 0.0
        }

        for keypoint, frames_padding_mask, embedding, mask_embedding, label in self.train_loader:
            # DEBUG
            if DEBUG and epoch == 0:
                with torch.no_grad():
                    mv = (~mask_embedding.bool()).sum(dim=1)  # #tokens válidos por muestra
                    tqdm.write(f"[DEBUG] valid_tokens_text: min={mv.min().item()}, mean={mv.float().mean().item():.2f}")

            if self.save_tb_model and epoch == 1 and not getattr(self, "graph_added", False):
                print("Saving graph")
                self.writer.add_graph(self.model, (keypoint, frames_padding_mask))
                self.graph_added = True           
            
            with self.accelerator.accumulate(self.model):
                # self.optimizer.zero_grad(set_to_none=True)
                train_loss, metrics = self._train_batch(keypoint, frames_padding_mask, embedding, mask_embedding, label)

                if DEBUG and not hasattr(self, "_lrchk"):
                    self.accelerator.print(f"[LR] {self.optimizer.param_groups[0]['lr']:.3e}")
                    self._lrchk = True


                total_loss += train_loss
                for k, v in metrics.items():
                    accumulated_metrics[k] += v
                
        # Calculate averages
        final_train_loss = total_loss.item()/len(self.train_loader)
        avg_metrics = {k: v/len(self.train_loader) for k, v in accumulated_metrics.items()}

        # Log to tensorboard
        self.writer.add_scalar("Loss/train", final_train_loss, epoch)
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"Metrics/{k}", v, epoch)

        if epoch % self.log_interval == 0:
            metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
            tqdm.write(f"\nEpoch: {epoch}.\n Train loss: {final_train_loss:.4f} {metrics_str}")

        return final_train_loss

    def _forward_loss(self, keypoint, frames_padding_mask, embedding, mask_embedding):
        with self.accelerator.autocast():
            _, pool_out = self.model(keypoint, frames_padding_mask)

            pool_out = F.normalize(torch.nan_to_num(pool_out), dim=-1)

            if ((~mask_embedding.bool()).sum(dim=1) == 0).any():
                idx = ((~mask_embedding.bool()).sum(dim=1) == 0).nonzero(as_tuple=True)[0]
                mask_embedding[idx, 0] = False
                        
            # DEBUG
            if DEBUG and not hasattr(self, "_dbg_val_text"):
                # conteo de tokens válidos
                tv = (~mask_embedding.bool()).sum(dim=1)
                # stats del embedding crudo SOLO en posiciones válidas
                m = ~mask_embedding.bool()
                emb_valid = embedding[m]  # [N_valid, D]
                self.accelerator.print(
                    f"[VAL-DEBUG] text_valid/min={tv.min().item()} mean={tv.float().mean().item():.2f} | "
                    f"emb_valid mean={emb_valid.float().mean().item():.4f} std={emb_valid.float().std().item():.4f}"
                )
                self._dbg_val_text = True
                
            text_emb = masked_mean_pool(embedding, mask_embedding, dim=1)
            text_emb = F.normalize(torch.nan_to_num(text_emb), dim=-1)
            
            # DEBUG
            if DEBUG and not hasattr(self, "_dbg_done"):
                tqdm.write(f"[DEBUG] ||pool|| mean={pool_out.norm(dim=1).mean().item():.3f}  "
                    f"||text|| mean={text_emb.norm(dim=1).mean().item():.3f}")
                self._dbg_done = True

            loss, metrics = self.criterion(pool_out, text_emb)            
        return loss, metrics

    @nvtx.annotate("Train: Train Batch", color="green")
    def _train_batch(self, keypoint, frames_padding_mask, embedding, mask_embedding, label):
        self.optimizer.zero_grad(set_to_none=True)

        embs_v, embs_t = [], []
        batch_size = keypoint.size(0)
        n_sub_batch = (batch_size + self.sub_batch - 1) // self.sub_batch if self.batch_sampling else 1

        for i in range(n_sub_batch):
            start = i * self.sub_batch if self.batch_sampling else 0
            end   = min(start + self.sub_batch, batch_size) if self.batch_sampling else batch_size

            with self.accelerator.autocast():
                _, v = self.model(keypoint[start:end], frames_padding_mask[start:end])

                t = masked_mean_pool(embedding[start:end], mask_embedding[start:end], dim=1)
                pre_norm = t.norm(dim=1)
                
                # DEBUG
                if DEBUG and not hasattr(self, "_val_t_once"):
                    self.accelerator.print(f"[VAL-CHK] text_emb pre-norm: min={pre_norm.min().item():.4e} mean={pre_norm.mean().item():.4e}")
                    self._val_t_once = True
            v = torch.nan_to_num(v)
            t = torch.nan_to_num(t)
            
            v = F.normalize(v, dim=-1)
            t = F.normalize(t, dim=-1)

            embs_v.append(v)
            embs_t.append(t)

        V = torch.cat(embs_v, dim=0)  # [B,D]
        T = torch.cat(embs_t, dim=0)  # [B,D]

        if DEBUG and not hasattr(self, "_dbg_once"):
            vf = (~frames_padding_mask.bool()).sum(dim=1)
            vt = (~mask_embedding.bool()).sum(dim=1)
            self.accelerator.print(f"[DEBUG] frames valid/min={vf.min().item()} mean={vf.float().mean().item():.2f} | "
                                f"text valid/min={vt.min().item()} mean={vt.float().mean().item():.2f}")
            self.accelerator.print(f"[DEBUG] ||v|| mean={V.norm(dim=1).mean().item():.3f} ||t|| mean={T.norm(dim=1).mean().item():.3f}")
            self._dbg_once = True
        
        label_ids = torch.as_tensor(label, dtype=torch.long, device=V.device)
        loss, metrics = self.criterion(V, T, labels=label_ids)

        self.accelerator.backward(loss)
        if DEBUG and not hasattr(self, "_gradchk"):
            with torch.no_grad():
                g = 0.0
                n = 0
                for n_, p in self.model.named_parameters():
                    if p.grad is not None:
                        g += p.grad.data.pow(2).sum().item()
                        n += 1
                self.accelerator.print(f"[GRAD] params_with_grad={n} ||grad||={g**0.5:.3e} "
                                    f"logit_scale_lr={self.optimizer.param_groups[1]['lr']:.2e} "
                                    f"logit_scale={self.criterion.logit_scale.item():.3f}")
            self._gradchk = True


        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        self.ema.update_parameters(self.model)

        if self.scheduler is not None:
            self.scheduler.step()  # por batch

        # logging
        batch_loss = loss.detach()
        acc_metrics = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in metrics.items()}
        return batch_loss, acc_metrics

    @nvtx.annotate("Validation Section", color="green")
    def _val(self, epoch):
        self.model.eval()
        val_loss = 0
        accumulated_metrics = {
            'acc_v2t': 0.0,
            'acc_t2v': 0.0,
            'temp': 0.0
        }

        for keypoint, frames_padding_mask, embedding, mask_embedding, labels in self.val_loader:
            loss, metrics = self._val_batch(keypoint, frames_padding_mask, embedding, mask_embedding, labels)
            val_loss += loss
            for k, v in metrics.items():
                accumulated_metrics[k] += v

        # Calculate averages
        final_val_loss = val_loss.item() / len(self.val_loader)
        avg_metrics = {k: v/len(self.val_loader) for k, v in accumulated_metrics.items()}

        # Log to tensorboard
        self.writer.add_scalar("Loss/val", final_val_loss, epoch)
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"Val_Metrics/{k}", v, epoch)

        if epoch % self.log_interval == 0:
            metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
            tqdm.write(f"Validation loss: {final_val_loss:.4f} {metrics_str}")

        self.early_stopping(final_val_loss)
        return final_val_loss
    
    @nvtx.annotate("Val: Validate Batch", color="green")
    def _val_batch(self, keypoint, frames_padding_mask, embedding, mask_embedding, labels):
        embs_v, embs_t = [], []
        batch_size = keypoint.size(0)
        n_sub_batch = (batch_size + self.sub_batch - 1) // self.sub_batch if self.batch_sampling else 1

        flag = True

        with torch.no_grad():
            for i in range(n_sub_batch):
                s = i * self.sub_batch if self.batch_sampling else 0
                e = min(s + self.sub_batch, batch_size) if self.batch_sampling else batch_size

                # Garantiza ≥1 frame válido
                pad_frames = frames_padding_mask[s:e].clone()
                if ((~pad_frames).sum(dim=1) == 0).any():
                    idx = ((~pad_frames).sum(dim=1) == 0).nonzero(as_tuple=True)[0]
                    pad_frames[idx, 0] = False
                _, v = self.model(keypoint[s:e], pad_frames)

                pad_text = mask_embedding[s:e].clone()
                if ((~pad_text).sum(dim=1) == 0).any():
                    idx = ((~pad_text).sum(dim=1) == 0).nonzero(as_tuple=True)[0]
                    pad_text[idx, 0] = False
                t = masked_mean_pool(embedding[s:e], pad_text, dim=1)

                v = torch.nan_to_num(v); t = torch.nan_to_num(t)
                v = F.normalize(v, dim=-1); t = F.normalize(t, dim=-1)

                embs_v.append(v)
                embs_t.append(t)

            V = torch.cat(embs_v, dim=0)
            T = torch.cat(embs_t, dim=0)

            labels_ids = torch.as_tensor(labels, dtype=torch.long, device=V.device)
            if DEBUG and flag:
                with torch.no_grad():
                    # supongamos que 'labels' es un vector [B] de IDs emparejados video↔texto
                    assert V.size(0) == T.size(0)
                    # diagnostiquemos “vecino más cercano comparte etiqueta”
                    S = (V @ T.t()).float()  # [B,B]
                    nn_t = S.argmax(dim=1)   # mejor texto para cada video
                    nn_v = S.argmax(dim=0)   # mejor video para cada texto
                    top1_v_shares_label = (labels_ids == labels_ids[nn_t]).float().mean().item()
                    top1_t_shares_label = (labels_ids == labels_ids[nn_v]).float().mean().item()
                    self.accelerator.print(f"[VAL-PAIR] share_label_v2t={top1_v_shares_label:.3f} share_label_t2v={top1_t_shares_label:.3f}")

                # DEBUG
                with torch.no_grad():
                    S = (V @ T.t()).float()
                    B = S.size(0)
                    diag = torch.diag(S)
                    off = S[~torch.eye(B, dtype=torch.bool, device=S.device)]
                    self.accelerator.print(
                        f"[VAL-SIM] diag_mean={diag.mean():.3f} off_mean={off.mean():.3f} "
                        f"diag_p10={diag.kthvalue(max(1, int(0.1*B))).values:.3f} "
                        f"off_p90={off.kthvalue(max(1, int(0.9*off.numel()))).values:.3f}"
                    )
                    flag = False

            if DEBUG and not hasattr(self, "_dbg_val_once"):
                vf = (~frames_padding_mask.bool()).sum(dim=1)
                vt = (~mask_embedding.bool()).sum(dim=1)
                self.accelerator.print(f"[VAL-DEBUG] frames valid/min={vf.min().item()} mean={vf.float().mean().item():.2f} | "
                                    f"text valid/min={vt.min().item()} mean={vt.float().mean().item():.2f}")
                self.accelerator.print(f"[VAL-DEBUG] ||v|| mean={V.norm(dim=1).mean().item():.3f} ||t|| mean={T.norm(dim=1).mean().item():.3f}")
                self._dbg_val_once = True

            loss, metrics = self.criterion(V, T, labels_ids)

        return loss.detach(), {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in metrics.items()}
