import torch
import torch.nn as nn
from .components import TransformerEncoderLayerRoPE
import torch.utils.checkpoint as checkpoint
import typing as t
import torch.nn.functional as F

class Imitator(nn.Module):
    def __init__(
        self,
        input_size: int = 111*2,
        hidden_size: int = 512,
        output_size: int = 3072,
        nhead: int = 8,
        ff_dim: int = 1024,
        n_layers: int = 2,
        max_seq_length: int = 301,
        encoder_dropout: float = 0.4,
        cross_attention_dropout: float = 0.4,
        pool_dim: int = 256,
        pool_strategy: t.Literal['cls', 'mean'] = 'mean'
    ):
        super().__init__()
        
        self.cfg = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "nhead": nhead,
            "ff_dim": ff_dim,
            "n_layers": n_layers,
            "max_seq_length": max_seq_length,
            "encoder_dropout": encoder_dropout,
            "cross_attention_dropout": cross_attention_dropout,
            "pool_dim": pool_dim,
            "pool_strategy": pool_strategy,
        }
        self.pool_strategy = pool_strategy

        # --- Bloque de entrada ---

        self.linear_feat = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2)
        )

        # linear sequencer
        self.conv1  = nn.Conv1d(hidden_size//2, pool_dim, kernel_size=3, padding=1)
        self.ln1    = nn.LayerNorm(pool_dim)
        self.act1   = nn.GELU()
        self.conv2  = nn.Conv1d(pool_dim, pool_dim, kernel_size=1)
        self.ln2    = nn.LayerNorm(pool_dim)
        self.act2   = nn.GELU()
    
        # Volvemos a hidden_size
        self.linear_hidden = nn.Linear(pool_dim, hidden_size)

        # Positional Encoding + Transformer
        encoder_layer    = TransformerEncoderLayerRoPE(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=encoder_dropout,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Proyección final por paso de tiempo
        self.proj = nn.Linear(hidden_size, output_size)

        self.token_queries = nn.Parameter(torch.randn(max_seq_length, output_size))  # [1, output_size]
        # Queries = E_tokens [n_tokens × B × d], Keys/Values = frames_repr [T' × B × d]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_size,
            num_heads=nhead,
            dropout=cross_attention_dropout,
            batch_first=True,
        )

        self.norm_attn = nn.LayerNorm(output_size)

        self.proj_final = nn.Sequential(
            nn.Linear(output_size, output_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_size * 2, output_size)
        )

    def _pool_tokens(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # tokens: [B, T_eff, D], mask: [B, T_eff] con True=PAD (ignorar)
        if self.pool_strategy == 'mean':
            if mask is not None:
                valid = (~mask).unsqueeze(-1).to(tokens.dtype)          # 1.0 en válidos
                num = (tokens * valid).sum(dim=1)
                den = valid.sum(dim=1).clamp_min(1e-6)
                v = num / den
            else:
                v = tokens.mean(dim=1)
        else:  # 'cls'
            v = tokens[:, 0, :]

        return F.normalize(v, dim=-1)  # mantiene grad
    
    @torch.compile(dynamic=True)
    def forward(self, x:torch.Tensor, frames_padding_mask:torch.Tensor=None) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        x: Tensor of frames
        frames_padding_mask: Bool Tensor [B, T] (True = padding = ignorar)
        returns: Tensor of embeddings for each token (128 tokens of frames)
        """

        def transformer_block(x):
            return self.transformer(x,src_key_padding_mask=frames_padding_mask)

        B, T, D, K = x.shape                # x -> [batch_size, T, input_size]
        x = x.view(B, T,  D * K)            # [B, T, input_size]
        
        x = self.linear_feat(x)             # [B, T, hidden//2]

        x = x.transpose(1, 2)               # [B, hidden//2, T]
        # se mantiene T' = T o reducirdo a pool_dim
        x  = self.conv1(x)                  # [B, hidden//2, pool_dim]
        x = x.transpose(1, 2)               # [B, pool_dim, hidden//2]
        x = self.ln1(x)                     # [B, pool_dim, hidden//2]
        x = self.act1(x)                    # [B, pool_dim, hidden//2]
        x = x.transpose(1, 2)               # [B, hidden//2, pool_dim]

        x = self.conv2(x)                  # [B, pool_dim, pool_dim]
        x = x.transpose(1, 2)               # [B, pool_dim, pool_dim]
        x = self.ln2(x)                     # [B, pool_dim, pool_dim]
        x = self.act2(x)                    # [B, pool_dim, pool_dim]

        x = self.linear_hidden(x)           # [B, pool_dim, hidden]

        pad = None
        if frames_padding_mask is not None:
            pad = frames_padding_mask.bool()  # True = ignorar
            # Garantiza al menos 1 válido por muestra
            valid_counts = (~pad).sum(dim=1)
            if (valid_counts == 0).any():
                idx = (valid_counts == 0).nonzero(as_tuple=True)[0]
                pad[idx, 0] = False
            pad = pad.contiguous()

        def enc(z):
            return self.transformer(z, src_key_padding_mask=pad)

        if self.training:
            x = checkpoint.checkpoint(enc, x, use_reentrant=True)
        else:
            x = enc(x)

        M = self.proj(x).contiguous()        # [B, pool_dim, output_size]
        M = torch.nan_to_num(M)
        
        T_eff = min(T, self.token_queries.shape[0])
        Q = self.token_queries[:T_eff].unsqueeze(0).expand(B, -1, -1).contiguous()
        M = M[:, :T_eff, :]
        pad_eff = pad[:, :T_eff] if pad is not None else None

        attn_out, _ = self.cross_attn(query=Q, key=M, value=M, key_padding_mask=pad_eff) # [B, n_tokens, output_size]
        attn_out = torch.nan_to_num(attn_out)
        
        x = self.norm_attn(Q + attn_out)
        # print(f"Attention output shape: {attn_out.shape}, Q shape: {Q.shape}, M shape: {M.shape}")
        x = x + self.proj_final(attn_out)        # [B, n_tokens, output_size]
        x = torch.nan_to_num(x)
        pooled = self._pool_tokens(x, mask=pad_eff) # [B, output_size]
        return x, pooled  # [B, n_tokens, output_size], [B, output_size]