import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipContrastiveLoss(nn.Module):
    def __init__(self, init_temp=0.07, max_scale=100.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temp)))
        self.max_scale = max_scale

    def forward(self, vid_emb: torch.Tensor, txt_emb: torch.Tensor, labels: torch.Tensor|None=None):
        with torch.no_grad():
            self.logit_scale.data.clamp_(max=torch.log(torch.tensor(self.max_scale)))

        # vid_emb, txt_emb: [B, D] (idealmente ya normalizados)
        v = F.normalize(vid_emb, dim=-1)
        t = F.normalize(txt_emb, dim=-1)

        scale = self.logit_scale.clamp(
            max=torch.log(torch.tensor(50.0, device=v.device, dtype=v.dtype))
        ).exp()
        logits_vt = scale * (v @ t.t())   # [B, B_total]
        logits_tv = scale * (t @ v.t())   # [B, B_total]

        if labels is not None:
            B = v.size(0)
            same = labels[:, None].eq(labels[None, :])       # [B,B]
            eye  = torch.eye(B, dtype=torch.bool, device=v.device)
            neg_same = same & ~eye                            # mismos labels ≠ diagonal
            neg_mask_val = torch.finfo(logits_vt.dtype).min   # “-inf” estable
            logits_vt = logits_vt.masked_fill(neg_same, neg_mask_val)
            logits_tv = logits_tv.masked_fill(neg_same, neg_mask_val)
            
        # Targets: índice del positivo
        targets = torch.arange(v.size(0), device=v.device)
        loss_v = F.cross_entropy(logits_vt, targets)
        loss_t = F.cross_entropy(logits_tv, targets)
        loss = 0.5 * (loss_v + loss_t)

        with torch.no_grad():
            acc_v = (logits_vt.argmax(dim=1) == targets).float().mean()
            acc_t = (logits_tv.argmax(dim=1) == targets).float().mean()
            temp = 1.0 / scale

        return loss, {"acc_v2t": acc_v, "acc_t2v": acc_t, "temp": temp}
