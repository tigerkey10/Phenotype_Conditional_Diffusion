"""Conditional VAE baseline on the same log-norm HVG input space."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ConditionalVAE(nn.Module):
    def __init__(self, n_genes: int, n_classes: int, latent_dim: int = 32, hidden: int = 512):
        super().__init__()
        self.n_classes = n_classes
        self.class_emb = nn.Embedding(n_classes, hidden)

        self.enc = nn.Sequential(
            nn.Linear(n_genes + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_genes),
        )
        self.latent_dim = latent_dim

    def encode(self, x, c):
        h = self.enc(torch.cat([x, self.class_emb(c)], dim=-1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, c):
        return self.dec(torch.cat([z, self.class_emb(c)], dim=-1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z, c), mu, logvar


def train_cvae(model, X, y, cfg, device):
    model.to(device).train()
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    kl_w = cfg["kl_weight"]
    for ep in range(cfg["epochs"]):
        tot = 0.0
        for xb, cb in dl:
            xb, cb = xb.to(device), cb.to(device)
            recon, mu, logvar = model(xb, cb)
            rec = F.mse_loss(recon, xb, reduction="mean")
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + kl_w * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
    return model


@torch.no_grad()
def sample_cvae(model, n: int, c_idx: int, device) -> np.ndarray:
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device)
    c = torch.full((n,), c_idx, device=device, dtype=torch.long)
    return model.decode(z, c).cpu().numpy()
