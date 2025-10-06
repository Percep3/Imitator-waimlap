import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipContrastiveLoss(nn.Module):
    def __init__(self, init_temp=0.07, max_scale=100.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temp)))
        self.max_scale = max_scale

    def forward(self, vid_emb: torch.Tensor, txt_emb: torch.Tensor):
        with torch.no_grad():
            self.logit_scale.data.clamp_(max=torch.log(torch.tensor(self.max_scale)))

        # vid_emb, txt_emb: [B, D] (idealmente ya normalizados)
        v = F.normalize(vid_emb, dim=-1)
        t = F.normalize(txt_emb, dim=-1)

        scale = self.logit_scale.exp()
        logits_v2t = scale * (v @ t.t())   # [B, B_total]
        logits_t2v = scale * (t @ v.t())   # [B, B_total]

        # Targets: índice del positivo
        targets = torch.arange(v.size(0), device=v.device)
        loss_v = F.cross_entropy(logits_v2t, targets)
        loss_t = F.cross_entropy(logits_t2v, targets)
        loss = 0.5 * (loss_v + loss_t)

        with torch.no_grad():
            acc_v = (logits_v2t.argmax(dim=1) == targets).float().mean()
            acc_t = (logits_t2v.argmax(dim=1) == targets).float().mean()
            temp = 1.0 / scale.item()

        return loss, {"acc_v2t": acc_v.item(), "acc_t2v": acc_t.item(), "temp": temp}
