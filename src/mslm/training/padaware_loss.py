import torch
import torch.nn as nn
import torch.nn.functional as F

class PadAwareTokenLoss(nn.Module):
    """
    pred_tokens: [B, T_pred, D]  (x devuelto por Imitator)
    text_tokens: [B, T_text, D]  (embeddings por token desde el LLM)
    text_pad_mask: [B, T_text]   (True = PAD)
    """
    def __init__(self,
                 lambda_token=0.2,      # alineación token a token (no-pad)
                 lambda_pad=0.5,        # pull/push con e_pad
                 lambda_mono=0.2,       # monotonía s_pad[i+1] >= s_pad[i]
                 lambda_boundary=0.2,   # salto nítido en frontera no-pad → pad
                 margin_push=0.2,       # empuje mínimo lejos de pad en no-pad
                 margin_boundary=0.3):  # gap mínimo entre frontera
        super().__init__()
        self.l_tok  = lambda_token
        self.l_pad  = lambda_pad
        self.l_mono = lambda_mono
        self.l_bnd  = lambda_boundary
        self.m_push = margin_push
        self.m_bnd  = margin_boundary

    def forward(self, pred_tokens, text_tokens, text_pad_mask):
        B, T_pred, D = pred_tokens.shape
        B2, T_text, D2 = text_tokens.shape
        assert B == B2 and D == D2, "Batch o dim no coinciden"

        T = min(T_pred, T_text)
        P = F.normalize(pred_tokens[:, :T, :], dim=-1)  # [B,T,D]
        E = F.normalize(text_tokens[:, :T, :], dim=-1)  # [B,T,D]
        M = text_pad_mask[:, :T]                        # [B,T] True=PAD

        # --- e_pad por batch: media de embeddings en posiciones de PAD ---
        # fallback: si una muestra no tiene PAD en los primeros T, usa el último token como proxy
        with torch.no_grad():
            has_pad = M.any(dim=1)  # [B]
        # evita NaN: usa mean con clamp en conteo
        pad_mask_f = (~M).float()  # válidos = 1.0, pad=0.0
        # Para e_pad, necesitamos la media SOLO en posiciones PAD. Creamos el complemento:
        pad_only = M.unsqueeze(-1).float()  # 1.0 en PAD, 0.0 en no-PAD
        denom = pad_only.sum(dim=1).clamp_min(1e-6)
        e_pad_batch = (E * pad_only).sum(dim=1) / denom  # [B,D]
        # para samples sin pad en esa ventana, usa el último embedding como aproximación estable
        last_tok = E[:, -1, :]
        e_pad_batch = torch.where(
            has_pad.unsqueeze(-1),
            e_pad_batch,
            last_tok
        )
        e_pad_batch = F.normalize(e_pad_batch, dim=-1)  # [B,D]

        # --- similitud con target token (solo no-pad) ---
        cos_tok = F.cosine_similarity(P, E, dim=-1)      # [B,T]
        if (~M).any():
            loss_token = (1.0 - cos_tok[~M]).mean()
        else:
            loss_token = P.new_tensor(0.0)

        # --- similitud con PAD ---
        e_pad_exp = e_pad_batch.unsqueeze(1).expand(-1, T, -1)  # [B,T,D]
        s_pad = F.cosine_similarity(P, e_pad_exp, dim=-1)       # [B,T] alto=se parece a PAD

        # Pull en PAD: queremos s_pad ~ 1 en posiciones PAD
        if M.any():
            loss_pad_pull = (1.0 - s_pad[M]).mean()
        else:
            loss_pad_pull = P.new_tensor(0.0)

        # Push en no-PAD: penaliza si s_pad es “demasiado alto” en no-pad
        # (queremos s_pad <= 1 - m_push)
        if (~M).any():
            loss_pad_push = F.relu(s_pad[~M] - (1.0 - self.m_push)).mean()
        else:
            loss_pad_push = P.new_tensor(0.0)

        # Monotonía: s_pad[i+1] >= s_pad[i]
        if T > 1:
            diff = s_pad[:, 1:] - s_pad[:, :-1]
            loss_mono = F.relu(-diff).mean()
        else:
            loss_mono = P.new_tensor(0.0)

        # Frontera: en transiciones no-pad→pad, queremos salto claro
        if T > 1:
            boundary = (~M[:, :-1]) & (M[:, 1:])  # True en i donde i es no-pad y i+1 es pad
            if boundary.any():
                s_prev = s_pad[:, :-1][boundary]
                s_next = s_pad[:, 1:][boundary]
                # exigir s_next >= s_prev + margin
                loss_boundary = F.relu((s_prev + self.m_bnd) - s_next).mean()
            else:
                loss_boundary = P.new_tensor(0.0)
        else:
            loss_boundary = P.new_tensor(0.0)

        loss = (self.l_tok  * loss_token +
                self.l_pad  * (loss_pad_pull + loss_pad_push) +
                self.l_mono * loss_mono +
                self.l_bnd  * loss_boundary)

        stats = dict(
            token=loss_token.detach(),
            pad_pull=loss_pad_pull.detach(),
            pad_push=loss_pad_push.detach(),
            mono=loss_mono.detach(),
            boundary=loss_boundary.detach(),
            s_pad_mean=s_pad.mean().detach()
        )
        return loss, stats
