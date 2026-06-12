"""Sampling and zero-inflation handling for the trained diffusion.

Provides: checkpoint loading (with the stored scaler / x0-clip bounds / dropout
rates / magnitude quantiles); class-conditional sampling; the two zero-gates that
restore scRNA-seq sparsity (hurdle = MLE-rate magnitude-ranked, learned = gate-head
probability-ranked); quantile magnitude calibration; and per-cell counterfactual
editing. See README §5 (design choices) and §7-§11 for the rationale behind each.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

from .diffusion import GaussianDiffusion
from .model import ResidualMLPDenoiser


def rank_gate(g: np.ndarray, score: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Zero out the `rate[g]` fraction of entries per gene with the lowest `score`,
    restoring scRNA-seq sparsity. Continuous diffusion cannot emit exact zeros, so
    it fills dropout positions with small positive noise; we model on/off as a
    per-class/gene Bernoulli whose probability is fit by MLE (the empirical zero
    rate). Ranking selects *which* cells stay on:
      - hurdle gate:  score = generated magnitude (keep highest-expression cells)
      - learned gate: score = gate-head P(expressed) (content-conditioned)

    g: (N, G) generated cells for one class.  score: (N, G).  rate: (G,) zeros.
    """
    N, G = g.shape
    out = g.copy()
    k = np.rint(rate * N).astype(int)            # cells to zero per gene
    for gi in range(G):
        kk = int(k[gi])
        if kk <= 0:
            continue
        if kk >= N:
            out[:, gi] = 0.0
            continue
        idx = np.argpartition(score[:, gi], kk - 1)[:kk]   # kk lowest-score cells
        out[idx, gi] = 0.0
    return out


def hurdle_gate(g: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Hurdle gate ranked by generated magnitude (see rank_gate)."""
    return rank_gate(g, g, rate)


def quantile_calibrate(g: np.ndarray, quant: np.ndarray) -> np.ndarray:
    """Map each gene's nonzero generated values, rank-preservingly, onto the real
    expressed-value distribution for that gene/class. Fixes the magnitude of
    expressed genes (diffusion compresses the dynamic range and undershoots),
    while preserving the diffusion's per-gene cell ordering and the cross-gene
    joint structure. `quant`: (G, Q) real nonzero quantiles at evenly spaced
    levels; the diffusion supplies which genes co-express, calibration supplies
    each gene's marginal scale.
    """
    N, G = g.shape
    Q = quant.shape[1]
    qpos = np.linspace(0.0, 1.0, Q)
    out = g.copy()
    for gi in range(G):
        col = g[:, gi]
        nz = np.nonzero(col > 0)[0]
        k = nz.size
        if k == 0 or quant[gi].max() == 0:
            continue
        order = nz[np.argsort(col[nz])]                  # ascending by gen value
        ranks = (np.arange(k) + 0.5) / k                 # fractional ranks in (0,1)
        out[order, gi] = np.interp(ranks, qpos, quant[gi]).astype(np.float32)
    return out


def load_model(ckpt_path: str, device) -> ResidualMLPDenoiser:
    ckpt = torch.load(ckpt_path, map_location=device)
    mc = ckpt["model_cfg"]
    model = ResidualMLPDenoiser(
        n_genes=ckpt["n_genes"], n_classes=ckpt["n_classes"],
        hidden=mc["hidden"], n_blocks=mc["n_blocks"],
        time_dim=mc["time_dim"], class_dim=mc["class_dim"], dropout=mc["dropout"],
        gate_head=mc.get("gate_head", False),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    model._n_genes = ckpt["n_genes"]
    model._n_classes = ckpt["n_classes"]
    # Standardization stats (None for legacy checkpoints) used to map generated
    # samples back to the log-norm space.
    model._scaler_mean = ckpt.get("scaler_mean")
    model._scaler_std = ckpt.get("scaler_std")
    lo, hi = ckpt.get("x0_lo"), ckpt.get("x0_hi")
    if lo is not None:
        model._x0_clip = (torch.from_numpy(lo).to(device), torch.from_numpy(hi).to(device))
    else:
        model._x0_clip = None
    model._zero_rate = ckpt.get("zero_rate")    # (n_classes, G) or None
    model._mag_quant = ckpt.get("mag_quant")    # (n_classes, G, Q) or None
    return model


@torch.no_grad()
def generate_cells(diffusion, model, n_genes, n, c_idx, steps, w, x0_clip, mean, std,
                   gate_mode, zero_rate, device, mag_quant=None) -> np.ndarray:
    """Generate n cells of one class, apply the zero-gate, then (optionally) the
    magnitude calibration.

    gate_mode: "learned" ranks cells by the gate head's P(expressed) and zeros the
    empirical-rate fraction with lowest probability (content-conditioned, exact
    sparsity); "hurdle" ranks by generated magnitude; "none" leaves the raw output.
    mag_quant: (G, Q) real nonzero quantiles for this class — if given, expressed
    values are calibrated to the real per-gene scale.
    """
    cond = torch.full((n,), c_idx, device=device, dtype=torch.long)
    x = diffusion.ddim_sample(model, n, n_genes, cond, steps=steps, w=w,
                              x0_clip=x0_clip)

    prob = None
    if gate_mode == "learned" and getattr(model, "gate_head", False):
        # Gate P(expressed) on the (nearly clean) generated x0 at t=0.
        t0 = torch.zeros(n, device=device, dtype=torch.long)
        prob = torch.sigmoid(model.gate_logits(x, t0, cond)).cpu().numpy()

    g = x.cpu().numpy().astype(np.float32)
    if mean is not None and std is not None:
        g = g * std + mean                     # invert standardization
    g = np.clip(g, 0.0, None)                  # log-norm expression is non-negative

    if prob is not None and zero_rate is not None:
        g = rank_gate(g, prob, zero_rate[c_idx])       # learned, prob-ranked
    elif gate_mode == "hurdle" and zero_rate is not None:
        g = hurdle_gate(g, zero_rate[c_idx])           # MLE-rate, magnitude-ranked

    if mag_quant is not None:
        g = quantile_calibrate(g, mag_quant)           # fix expressed magnitude
    return g


@torch.no_grad()
def counterfactual_edit(diffusion, model, X_real, c_target, strength, steps, w,
                        x0_clip, mean, std, gate_mode, zero_rate, device,
                        mag_quant=None) -> np.ndarray:
    """Per-cell counterfactual: edit real cells X_real (eval space, log-norm) toward
    class c_target at the given noise `strength`, preserving per-cell structure.
    Returns edited cells in eval space."""
    n = X_real.shape[0]
    xs = (X_real - mean) / std if mean is not None else X_real
    x0 = torch.from_numpy(xs.astype(np.float32)).to(device)
    cond = torch.full((n,), c_target, device=device, dtype=torch.long)
    x = diffusion.ddim_edit(model, x0, cond, strength, steps=steps, w=w, x0_clip=x0_clip)

    prob = None
    if gate_mode == "learned" and getattr(model, "gate_head", False):
        t0 = torch.zeros(n, device=device, dtype=torch.long)
        prob = torch.sigmoid(model.gate_logits(x, t0, cond)).cpu().numpy()

    g = x.cpu().numpy().astype(np.float32)
    if mean is not None and std is not None:
        g = g * std + mean
    g = np.clip(g, 0.0, None)
    if prob is not None and zero_rate is not None:
        g = rank_gate(g, prob, zero_rate[c_target])
    elif gate_mode == "hurdle" and zero_rate is not None:
        g = hurdle_gate(g, zero_rate[c_target])
    if mag_quant is not None:
        g = quantile_calibrate(g, mag_quant)
    return g


@torch.no_grad()
def sample_all_classes(cfg, model, device, w: float) -> Dict[int, np.ndarray]:
    """Return {class_idx: array(n_per_class, n_genes)} at guidance scale w."""
    diffusion = GaussianDiffusion(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        schedule=cfg["diffusion"]["schedule"],
        pred_type=cfg["diffusion"].get("pred_type", "x0"), device=device,
    )
    n = cfg["sample"]["n_per_class"]
    gate = cfg["sample"].get("zero_gate", "hurdle")
    calib = cfg["sample"].get("magnitude_calibration", "none") == "quantile"
    mq = getattr(model, "_mag_quant", None)
    return {
        c: generate_cells(
            diffusion, model, model._n_genes, n, c, cfg["sample"]["ddim_steps"], w,
            getattr(model, "_x0_clip", None), model._scaler_mean, model._scaler_std,
            gate, getattr(model, "_zero_rate", None), device,
            mag_quant=(mq[c] if (calib and mq is not None) else None),
        )
        for c in range(model._n_classes)
    }


def save_samples(cfg, samples: Dict[int, np.ndarray], w: float) -> str:
    out_dir = os.path.join(cfg["out_dir"], "samples")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ddpm_w{w}.npz")
    np.savez_compressed(path, **{f"class_{c}": v for c, v in samples.items()})
    return path
