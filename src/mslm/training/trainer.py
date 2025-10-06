from tqdm import tqdm
import typing as t

from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
import torch
import random

torch.manual_seed(23)
random.seed(23)

from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from src.mslm.utils.early_stopping import EarlyStopping
from src.mslm.checkpoint.manager import CheckpointManager
# from src.mslm.training import imitator_loss
from src.mslm.training.clip_loss import ClipContrastiveLoss
import nvtx
from datetime import datetime

def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x:    [B, T, D]
    mask: [B, T] con True = válido
    """
    mask = mask.unsqueeze(-1).to(x.dtype)  # [B, T, 1]
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(1e-8)
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
                
        #Dataloaders
        self.train_loader = self.accelerator.prepare_data_loader(train_loader)
        self.val_loader = self.accelerator.prepare_data_loader(val_loader)

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
                {"params": [self.criterion.logit_scale], "weight_decay": 0.0},
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

        warmup_steps = 5 * len(self.train_loader)  # p.ej. 5 epochs de warm-up
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

        for keypoint, frames_padding_mask, embedding, mask_embedding in self.train_loader:
            # DEBUG
            if epoch == 0:
                with torch.no_grad():
                    mv = (~mask_embedding.bool()).sum(dim=1)  # #tokens válidos por muestra
                    tqdm.write(f"[DEBUG] valid_tokens_text: min={mv.min().item()}, mean={mv.float().mean().item():.2f}")

            if self.save_tb_model and epoch == 1 and not getattr(self, "graph_added", False):
                print("Saving graph")
                self.writer.add_graph(self.model, (keypoint, frames_padding_mask))
                self.graph_added = True           
            
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad(set_to_none=True)        
                train_loss, metrics = self._train_batch(keypoint, frames_padding_mask, embedding, mask_embedding)

                if self.scheduler is not None:
                    self.scheduler.step()

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
            output, pool_out = self.model(keypoint, frames_padding_mask)
            mask_valid = ~mask_embedding.bool()   # True = válido
            if (mask_valid.sum(dim=1) == 0).any():
                # si alguna fila quedó sin válidos, se hace todo válido en esa fila
                fix = (mask_valid.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
                mask_valid[fix] = True
                
            text_emb = masked_mean_pool(embedding, mask_valid, dim=1)
            
            # DEBUG
            if not hasattr(self, "_dbg_done"):
                tqdm.write(f"[DEBUG] ||pool|| mean={pool_out.norm(dim=1).mean().item():.3f}  "
                    f"||text|| mean={text_emb.norm(dim=1).mean().item():.3f}")
                self._dbg_done = True

            loss, metrics = self.criterion(pool_out, text_emb)            
        return loss, metrics

    @nvtx.annotate("Train: Train Batch", color="green")
    def _train_batch(self, keypoint, frames_padding_mask, embedding, mask_embedding):
        self.optimizer.zero_grad(set_to_none=True)

        embs_v, embs_t = [], []
        batch_size = keypoint.size(0)
        n_sub_batch = (batch_size + self.sub_batch - 1) // self.sub_batch if self.batch_sampling else 1

        for i in range(n_sub_batch):
            start = i * self.sub_batch if self.batch_sampling else 0
            end   = min(start + self.sub_batch, batch_size) if self.batch_sampling else batch_size

            with self.accelerator.autocast():
                _, v = self.model(keypoint[start:end], frames_padding_mask[start:end])

                mask_valid = ~mask_embedding[start:end].bool()
                if (mask_valid.sum(dim=1) == 0).any():
                    fix = (mask_valid.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
                    mask_valid[fix] = True
                t = masked_mean_pool(embedding[start:end], mask_valid, dim=1)

            embs_v.append(v)
            embs_t.append(t)

        V = torch.cat(embs_v, dim=0)  # [B,D]
        T = torch.cat(embs_t, dim=0)  # [B,D]

        loss, metrics = self.criterion(V, T)

        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
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

        for keypoint, frames_padding_mask, embedding, mask_embedding in self.val_loader:        
            loss, metrics = self._val_batch(keypoint, frames_padding_mask, embedding, mask_embedding)
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
    def _val_batch(self, keypoint, frames_padding_mask, embedding, mask_embedding) -> t.Tuple[torch.Tensor, dict]:
        batch_loss = 0.0
        accumulated_metrics = {
            'acc_v2t': 0.0,
            'acc_t2v': 0.0,
            'temp': 0.0
        }
        
        batch_size = keypoint.size(0)
        start = 0
        end = keypoint.size(0)
        
        # Si batch_sampling es False, procesamos todo el batch de una vez
        n_sub_batch = 1
        if self.batch_sampling:
            n_sub_batch = (batch_size + self.sub_batch - 1) // self.sub_batch

        with nvtx.annotate("Val: Forward + Loss", color="blue"):
            for i in range(n_sub_batch):
                # Actualizar índices solo si estamos haciendo batch sampling
                if self.batch_sampling:
                    start = i * self.sub_batch
                    end = min(start + self.sub_batch, batch_size)
                
                with nvtx.annotate("Forward Pass", color="blue"):
                    with torch.no_grad():
                        loss, metrics = self._forward_loss(keypoint[start:end], 
                                                    frames_padding_mask[start:end], 
                                                    embedding[start:end], 
                                                    mask_embedding[start:end])
                
                # Normalizar pérdida y métricas si estamos haciendo batch sampling
                if self.batch_sampling:
                    loss /= n_sub_batch
                    for k in metrics:
                        metrics[k] /= n_sub_batch
                        
                batch_loss += loss.detach()
                for k, v in metrics.items():
                    accumulated_metrics[k] += v

        return batch_loss, accumulated_metrics