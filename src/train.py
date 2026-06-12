"""Training loop for the 1D conditional DDPM.

AdamW (lr 1e-4, batch 256), 200 epochs, CFG label dropout p=0.1, EMA weights.
Before training it fits and stores everything the sampler needs in the checkpoint:
per-gene standardization (all-cell or expressed-only, `standardize_mode`), per-gene
x0-clip bounds, per-class dropout rates (hurdle gate) and nonzero quantiles
(calibration). With `gate_head` it also trains the dropout-gate BCE head and a
nonzero-weighted magnitude loss. Validation generates with the SAME gate/calibration
as deployment and keeps the lowest-mean-per-class-MMD EMA checkpoint.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .diffusion import GaussianDiffusion
from .evaluate import mmd_rbf
from .model import ResidualMLPDenoiser
from .sample import generate_cells
from .utils import EMA, ensure_dir


def _validate_mmd(diffusion, model, val_X, val_y, n_classes, steps, device,
                  mean, std, x0_clip, zero_rate, gate_mode, mag_quant=None, seed=0):
    """Mean per-class MMD between gated samples (w=1) and validation cells.

    Uses the same generation+gate path as final sampling so the best-checkpoint
    criterion matches the deployed sampler.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for c in range(n_classes):
        real = val_X[val_y == c]
        if real.shape[0] < 2:
            continue
        n = min(real.shape[0], 256)
        gen = generate_cells(diffusion, model, val_X.shape[1], n, c, steps, 1.0,
                             x0_clip, mean, std, gate_mode, zero_rate, device,
                             mag_quant=(mag_quant[c] if mag_quant is not None else None))
        r = real[rng.choice(real.shape[0], n, replace=False)] if real.shape[0] > n else real
        scores.append(mmd_rbf(r, gen))
    return float(np.mean(scores)) if scores else float("inf")


def train(cfg: dict, adata, train_idx, val_idx, device) -> Dict[str, str]:
    X = np.asarray(adata.X, dtype=np.float32)
    y = adata.obs["label"].values.astype(np.int64)
    n_genes = X.shape[1]
    n_classes = int(y.max() + 1)

    tr_X, tr_y = X[train_idx], y[train_idx]
    val_X, val_y = X[val_idx], y[val_idx]

    # Per-gene standardization (z-score) fit on the training split. Diffusion
    # assumes ~unit-variance inputs; log-norm scRNA-seq has tiny per-gene
    # variance and is ~93% zeros, so without this the signal is drowned by the
    # unit-variance forward noise. Stats are stored and inverted at sampling.
    # Per-class, per-gene dropout (zero) rate — MLE of the hurdle model's
    # on/off Bernoulli — computed on the raw log-norm data before scaling.
    zero_rate = np.stack([
        (tr_X[tr_y == c] == 0).mean(0) if (tr_y == c).any() else np.zeros(n_genes)
        for c in range(n_classes)
    ]).astype(np.float32)

    # Per-class/gene quantiles of real expressed (nonzero) values, for the
    # magnitude calibration that fixes the diffusion's compressed dynamic range.
    Q = cfg["sample"].get("calib_quantiles", 64)
    qpos = np.linspace(0.0, 1.0, Q)
    mag_quant = np.zeros((n_classes, n_genes, Q), dtype=np.float32)
    for c in range(n_classes):
        Xc = tr_X[tr_y == c]
        for gi in range(n_genes):
            nz = Xc[:, gi][Xc[:, gi] > 0]
            if nz.size:
                mag_quant[c, gi] = np.quantile(nz, qpos)

    tr_mask = (tr_X > 0).astype(np.float32)   # expressed-gene indicator for the gate

    # Standardization mode:
    #  allcell  (v1): mean/std over all cells — the 93%-zero spike dominates sigma, so
    #                 expressed values get squashed into a narrow range (the diffusion then
    #                 compresses them further, hence the v1 magnitude undershoot).
    #  expressed (v2/end-to-end): mean/std over expressed (nonzero) cells only, so expressed
    #                 values occupy a proper unit-variance range and the diffusion can
    #                 reproduce their magnitude directly — no post-hoc calibration needed.
    std_mode = cfg["train"].get("standardize_mode", "allcell")
    if std_mode == "expressed":
        g_mean = np.zeros(n_genes, np.float32)
        g_std = np.ones(n_genes, np.float32)
        for gi in range(n_genes):
            e = tr_X[:, gi][tr_X[:, gi] > 0]
            if e.size >= 2:
                g_mean[gi] = e.mean(); g_std[gi] = e.std()
            elif e.size == 1:
                g_mean[gi] = float(e[0])
        g_std = np.maximum(g_std, 0.5)        # floor: keep dropout encoding bounded
    else:
        g_mean = tr_X.mean(0)
        g_std = tr_X.std(0) + 1e-8
    tr_X = (tr_X - g_mean) / g_std

    # Per-gene clip bounds (standardized space, small margin) for stable DDIM.
    g_lo = tr_X.min(0)
    g_hi = tr_X.max(0)
    margin = 0.05 * (g_hi - g_lo)
    g_lo = (g_lo - margin).astype(np.float32)
    g_hi = (g_hi + margin).astype(np.float32)
    x0_clip = (torch.from_numpy(g_lo).to(device), torch.from_numpy(g_hi).to(device))

    ds = TensorDataset(torch.from_numpy(tr_X), torch.from_numpy(tr_y),
                       torch.from_numpy(tr_mask))
    dl = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        drop_last=False, num_workers=cfg["train"].get("num_workers", 0),
    )

    gate_mode = cfg["sample"].get("zero_gate", "hurdle")
    calib_mq = mag_quant if cfg["sample"].get("magnitude_calibration", "none") == "quantile" else None
    use_gate_head = gate_mode == "learned"
    cfg["model"]["gate_head"] = use_gate_head     # persisted in checkpoint meta
    gate_weight = cfg["train"].get("gate_weight", 1.0) if use_gate_head else 0.0
    nonzero_weight = cfg["train"].get("nonzero_weight", 1.0)

    model = ResidualMLPDenoiser(
        n_genes=n_genes, n_classes=n_classes,
        hidden=cfg["model"]["hidden"], n_blocks=cfg["model"]["n_blocks"],
        time_dim=cfg["model"]["time_dim"], class_dim=cfg["model"]["class_dim"],
        dropout=cfg["model"]["dropout"], gate_head=use_gate_head,
    ).to(device)
    ema = EMA(model, decay=cfg["train"]["ema_decay"])

    diffusion = GaussianDiffusion(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        schedule=cfg["diffusion"]["schedule"],
        pred_type=cfg["diffusion"].get("pred_type", "x0"),
        device=device,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    ckpt_dir = ensure_dir(os.path.join(cfg["out_dir"], "checkpoints"))
    best_path = os.path.join(ckpt_dir, "ddpm_best.pt")
    last_path = os.path.join(ckpt_dir, "ddpm_last.pt")
    best_mmd = float("inf")

    meta = {"n_genes": n_genes, "n_classes": n_classes, "model_cfg": cfg["model"],
            "scaler_mean": g_mean.astype(np.float32),
            "scaler_std": g_std.astype(np.float32),
            "x0_lo": g_lo, "x0_hi": g_hi, "zero_rate": zero_rate,
            "mag_quant": mag_quant}

    for ep in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = run_mag = run_gate = 0.0
        for xb, cb, mb in dl:
            xb, cb, mb = xb.to(device), cb.to(device), mb.to(device)
            loss, mag, gate = diffusion.p_losses(
                model, xb, cb, cfg["train"]["cfg_dropout"],
                mask=mb, gate_weight=gate_weight, nonzero_weight=nonzero_weight,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model)
            running += loss.item() * xb.size(0)
            run_mag += mag.item() * xb.size(0)
            run_gate += gate.item() * xb.size(0)
        running /= len(ds); run_mag /= len(ds); run_gate /= len(ds)

        if ep % cfg["train"]["eval_every"] == 0 or ep == cfg["train"]["epochs"]:
            val_mmd = _validate_mmd(
                diffusion, ema.module(), val_X, val_y, n_classes,
                cfg["sample"]["ddim_steps"], device, g_mean, g_std,
                x0_clip, zero_rate, gate_mode, mag_quant=calib_mq, seed=cfg["seed"],
            )
            tag = ""
            if val_mmd < best_mmd:
                best_mmd = val_mmd
                torch.save({"model": ema.module().state_dict(), **meta,
                            "val_mmd": val_mmd, "epoch": ep}, best_path)
                tag = "  <-- best"
            extra = f"  (mag {run_mag:.4f} gate {run_gate:.4f})" if use_gate_head else ""
            print(f"[epoch {ep:3d}] loss {running:.4f}{extra}  val_mmd {val_mmd:.4f}{tag}", flush=True)
        else:
            print(f"[epoch {ep:3d}] loss {running:.4f}", flush=True)

    torch.save({"model": ema.module().state_dict(), **meta,
                "val_mmd": best_mmd, "epoch": cfg["train"]["epochs"]}, last_path)
    if not os.path.exists(best_path):
        torch.save({"model": ema.module().state_dict(), **meta}, best_path)
    print(f"best val MMD = {best_mmd:.4f}  ->  {best_path}", flush=True)
    return {"best": best_path, "last": last_path}
