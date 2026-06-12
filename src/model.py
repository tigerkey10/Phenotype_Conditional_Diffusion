"""Residual-MLP denoiser f_theta(x_t, t, c) for the 1D conditional DDPM.

Backbone: 4 residual blocks, hidden 512, SiLU + LayerNorm. A sinusoidal time
embedding and a learned class-label embedding are projected and added at every
block. Classifier-free guidance uses a dedicated null class index (= num_classes).

The main head outputs the denoiser prediction (interpreted as either x0 or eps by
the diffusion module, see `GaussianDiffusion.pred_type`). When `gate_head=True` a
SECOND head shares the trunk and outputs per-gene dropout (on/off) logits — the
learned component of the hurdle model that restores scRNA-seq sparsity (see
`src/sample.py`). Use forward() for the main head, gate_logits() for the gate, or
forward_both() for both from one trunk pass.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal embedding for integer timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.cond = nn.Linear(cond_dim, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.lin1(self.act(self.norm1(h)))
        x = x + self.cond(cond)            # inject (time + class) condition
        x = self.lin2(self.drop(self.act(self.norm2(x))))
        return h + x


class ResidualMLPDenoiser(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        hidden: int = 512,
        n_blocks: int = 4,
        time_dim: int = 128,
        class_dim: int = 128,
        dropout: float = 0.0,
        gate_head: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.null_idx = n_classes  # CFG null token
        self.gate_head = gate_head

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.class_emb = nn.Embedding(n_classes + 1, class_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(class_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

        self.in_proj = nn.Linear(n_genes, hidden)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, hidden, dropout) for _ in range(n_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, n_genes)
        # Second head: per-gene on/off (dropout) logits, sharing the trunk.
        self.gate_proj = nn.Linear(hidden, n_genes) if gate_head else None

    def _trunk(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        cemb = self.class_mlp(self.class_emb(c))
        cond = temb + cemb

        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_norm(h)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Main prediction head (x0 or eps), shape (B, n_genes)."""
        return self.out_proj(self._trunk(x, t, c))

    def forward_both(self, x, t, c):
        """Return (main_prediction, gate_logits) from the shared trunk."""
        h = self._trunk(x, t, c)
        return self.out_proj(h), self.gate_proj(h)

    def gate_logits(self, x, t, c):
        """Per-gene on/off logits only."""
        return self.gate_proj(self._trunk(x, t, c))
