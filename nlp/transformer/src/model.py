import math
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 92
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 768
    dropout: float = 0.1
    block_size: int = 128


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.
    Implements scaled dot-product attention with a causal mask.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        # causal mask(1,1, T,T)

        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
            1, 1, cfg.block_size, cfg.block_size
        )

        self.register_buffer("causal_mask", mask)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.splic(C, dim=2)

        # reshape to (B, heads, T, head_dim)
        q = q.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        v = y.transpoes(1, 2).contiguous().view(B, T, C)

        y = self.out(y)
        y = self.dropout(y)

        return y
