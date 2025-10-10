import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenSoftAlignLoss(nn.Module):
    """
    pred_tokens: [B, T_pred, D]  (x que devuelve Imitator)
    text_tokens: [B, T_text, D]  (embeddings por token del LLM)
    text_pad_mask: [B, T_text]   (True = PAD)
    """
    def __init__(self, tau=5.0, entropy_reg=0.0):
        super().__init__()
        self.tau = tau
        self.entropy_reg = entropy_reg

    def forward(self, pred_tokens, text_tokens, text_pad_mask):
        B, Tp, D = pred_tokens.shape
        _, Tt, _ = text_tokens.shape
        T = min(Tp, Tt)

        P = F.normalize(pred_tokens[:, :T, :], dim=-1)  # [B,T,D]
        E = F.normalize(text_tokens[:, :T, :], dim=-1)  # [B,T,D]
        M = text_pad_mask[:, :T]                        # [B,T] True=PAD

        # similitud P_i vs E_j del MISMO par
        S = torch.einsum("btd,bsd->bts", P, E)          # [B,T,T]
        valid_cols = (~M).unsqueeze(1).expand(-1, T, -1)  # [B,T,T]
        S = S.masked_fill(~valid_cols, -1e9)

        # soft "matching" y objetivo suave
        A = torch.softmax(S * self.tau, dim=-1)         # [B,T,T]
        E_hat = torch.einsum("bts,bsd->btd", A, E)      # [B,T,D]

        # foca en filas con algún target válido
        row_has_valid = valid_cols.any(dim=-1)          # [B,T]
        if row_has_valid.any():
            cos = F.cosine_similarity(P[row_has_valid], E_hat[row_has_valid], dim=-1)
            loss = (1.0 - cos).mean()
            if self.entropy_reg > 0:
                A_rows = A[row_has_valid].clamp_min(1e-12)
                H = -(A_rows * A_rows.log()).sum(dim=-1).mean()
                loss = loss + self.entropy_reg * H
            tas = 1.0 - loss.detach()  # aprox (solo para logging)
        else:
            loss = P.new_tensor(0.0)
            tas = loss

        return loss, {"TAS_mean": tas}
