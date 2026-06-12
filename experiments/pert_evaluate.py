"""Kang perturbation benchmark — evaluate held-out perturbation prediction.

Predict the held-out cell type's STIM state from its CTRL cells, three ways:
  - ctrl     : no-op (predict ctrl unchanged) — lower bound
  - scGen    : VAE latent arithmetic (mean ctrl->stim shift) — the standard baseline
  - diffusion: per-cell counterfactual edit (raw SDEdit, structure-preserving)
and score against the real held-out stim cells: R^2 of mean expression (all genes and
the top perturbation DEGs), MMD, and recovery of the interferon signature (ISG15 etc.).

Run:  python experiments/pert_evaluate.py [holdout_celltype] [edit_strength]
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import anndata as ad
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import r2_score

from src.diffusion import GaussianDiffusion
from src.evaluate import mmd_rbf
from src.sample import load_model, counterfactual_edit
from src.utils import get_device, set_seed

HOLDOUT = sys.argv[1] if len(sys.argv) > 1 else "CD4 T cells"
STRENGTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
CACHE = "runs/outputs_kang/kang_hvg.h5ad"
ISGS = ["ISG15", "ISG20", "IFIT3", "IFIT2", "CXCL10", "RSAD2", "IFITM3"]


# ----------------------------- scGen VAE -----------------------------------
class VAE(nn.Module):
    def __init__(self, d, h=512, z=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.mu = nn.Linear(h, z); self.lv = nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, d))

    def encode(self, x):
        h = self.enc(x); return self.mu(h), self.lv(h)

    def forward(self, x):
        mu, lv = self.encode(x); z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv


def train_scgen_vae(X, device, epochs=200, bs=256, lr=1e-3):
    m = VAE(X.shape[1]).to(device).train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.from_numpy(X).float()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            xb = Xt[perm[i:i + bs]].to(device)
            rec, mu, lv = m(xb)
            loss = ((rec - xb) ** 2).mean() + 1e-3 * (-0.5 * (1 + lv - mu**2 - lv.exp()).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def scgen_predict(m, X_train, cond_train, ctrlX, device):
    """Latent delta (mean stim - mean ctrl over training) applied to held-out ctrl."""
    Z = m.encode(torch.from_numpy(X_train).float().to(device))[0].cpu().numpy()
    delta = Z[cond_train == 1].mean(0) - Z[cond_train == 0].mean(0)
    zc = m.encode(torch.from_numpy(ctrlX).float().to(device))[0]
    zc = zc + torch.from_numpy(delta).float().to(device)
    return m.dec(zc).cpu().numpy()


# ----------------------------- metrics -------------------------------------
def report(name, pred, real_stim, ctrl_mean, deg_idx, var_names):
    pm, rm = pred.mean(0), real_stim.mean(0)
    r2_all = r2_score(rm, pm)
    r2_deg = r2_score(rm[deg_idx], pm[deg_idx])
    # delta R^2: how well the *change* (stim-ctrl) is predicted on DEGs
    r2_delta = r2_score(rm[deg_idx] - ctrl_mean[deg_idx], pm[deg_idx] - ctrl_mean[deg_idx])
    sub = np.random.default_rng(0).choice(len(pred), min(500, len(pred)), replace=False)
    rsub = np.random.default_rng(1).choice(len(real_stim), min(500, len(real_stim)), replace=False)
    mmd = mmd_rbf(pred[sub], real_stim[rsub])
    print(f"{name:>12}  R2_all {r2_all:6.3f}  R2_DEG {r2_deg:6.3f}  R2_ΔDEG {r2_delta:6.3f}  MMD {mmd:6.3f}")
    return pm


def main():
    set_seed(0)
    device = get_device("cuda")
    adata = ad.read_h5ad(CACHE)
    X = np.asarray(adata.X, np.float32)
    y = adata.obs["label"].values.astype(int)
    ct = adata.obs["cell_type"].astype(str).values
    var_names = list(adata.var_names)

    ho_ctrl = (ct == HOLDOUT) & (y == 0)   # held-out type's control cells (the INPUT)
    ho_stim = (ct == HOLDOUT) & (y == 1)   # held-out type's stim cells (the TARGET, unseen in train)
    train_mask = ~ho_stim                  # train on everything except the held-out stim
    ctrlX = X[ho_ctrl]
    real_stim = X[ho_stim]
    ctrl_mean = ctrlX.mean(0)
    print(f"holdout {HOLDOUT}: {ho_ctrl.sum()} ctrl, {ho_stim.sum()} real stim (target)\n")

    # Perturbation DEGs: genes that move most ctrl->stim, measured on the OTHER cell
    # types only (the held-out type is never used to define them). R2 on these 50
    # genes is the meaningful score; R2 over all genes is trivially high (most are flat).
    tr = train_mask & (ct != HOLDOUT)
    diff = np.abs(X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0))
    deg_idx = np.argsort(diff)[::-1][:50]

    print(f"{'method':>12}  {'R2_all':>6}  {'R2_DEG':>6}  {'R2_ΔDEG':>7}  {'MMD':>6}")
    # 1) ctrl baseline
    report("ctrl(no-op)", ctrlX, real_stim, ctrl_mean, deg_idx, var_names)

    # 2) scGen latent arithmetic
    vae = train_scgen_vae(X[train_mask], device)
    pred_sg = scgen_predict(vae, X[train_mask], y[train_mask], ctrlX, device)
    sg_mean = report("scGen", pred_sg, real_stim, ctrl_mean, deg_idx, var_names)

    # 3) diffusion counterfactual (raw SDEdit, structure-preserving)
    cfg = yaml.safe_load(open("configs/default.yaml"))
    model = load_model("runs/outputs_kang/checkpoints/ddpm_best.pt", device)
    diff = GaussianDiffusion(timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"], beta_end=cfg["diffusion"]["beta_end"],
        schedule=cfg["diffusion"]["schedule"], pred_type=cfg["diffusion"]["pred_type"], device=device)
    # hurdle gate (restore stim sparsity), NO calibration (keep the ctrl->stim shift
    # the edit produces rather than overwriting it with generic-stim marginals).
    pred_df = counterfactual_edit(diff, model, ctrlX, 1, STRENGTH, cfg["sample"]["ddim_steps"],
        1.0, getattr(model, "_x0_clip", None), model._scaler_mean, model._scaler_std,
        "hurdle", getattr(model, "_zero_rate", None), device, mag_quant=None)
    df_mean = report(f"diffusion(s={STRENGTH})", pred_df, real_stim, ctrl_mean, deg_idx, var_names)

    # interferon signature recovery (mean log-expr): ctrl -> {pred} -> real
    print(f"\nISG recovery (mean expr):  {'gene':>8} {'ctrl':>6} {'scGen':>6} {'diff':>6} {'REAL':>6}")
    for g in ISGS:
        if g in var_names:
            gi = var_names.index(g)
            print(f"{'':>27}{g:>8} {ctrl_mean[gi]:>6.2f} {sg_mean[gi]:>6.2f} {df_mean[gi]:>6.2f} {real_stim.mean(0)[gi]:>6.2f}")


if __name__ == "__main__":
    main()
