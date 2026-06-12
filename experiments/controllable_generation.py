"""Diffusion-native controllable generation — capabilities scVI lacks.

Part A (guidance control): classifier-free guidance scale w is a sampling-time knob
that trades phenotype *prototypicality* for *diversity*, with no retraining. We score
generated cells with an oracle cell-type classifier (trained on all real cells) and
show purity/confidence rise and diversity falls as w increases.

Run:  python experiments/controllable_generation.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import data as datamod
from src.diffusion import GaussianDiffusion
from src.sample import load_model, counterfactual_edit
from src.utils import get_device


def oracle_classifier(X, y):
    """Strong cell-type judge trained on all real cells."""
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, C=1.0))
    clf.fit(X, y)
    return clf


def mean_pairwise_dist(A, n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(A.shape[0], min(n, A.shape[0]), replace=False)
    B = A[idx]
    d = np.sqrt(((B[:, None] - B[None]) ** 2).sum(-1))
    iu = np.triu_indices(len(B), 1)
    return float(d[iu].mean())


def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    adata = datamod.preprocess(cfg)
    X = np.asarray(adata.X, np.float32)
    y = adata.obs["label"].values.astype(int)
    classes = datamod.classes_of(adata)
    n_classes = len(classes)
    clf = oracle_classifier(X, y)
    print(f"oracle classifier train accuracy: {clf.score(X, y):.3f}\n")

    scales = cfg["sample"]["guidance_scales"]
    print("GUIDANCE CONTROL (v1 model) — purity = % generated cells classified as target")
    print(f"{'w':>4} {'purity':>8} {'confidence':>11} {'diversity':>10}")
    for w in scales:
        path = f"runs/outputs_v1_calibrated/samples/ddpm_w{float(w)}.npz"
        if not os.path.exists(path):
            continue
        d = np.load(path)
        purity, conf, div = [], [], []
        for c in range(n_classes):
            g = d[f"class_{c}"]
            proba = clf.predict_proba(g)
            pred = proba.argmax(1)
            purity.append((pred == c).mean())
            conf.append(proba[:, c].mean())
            div.append(mean_pairwise_dist(g))
        print(f"{float(w):>4} {np.mean(purity):>8.3f} {np.mean(conf):>11.3f} {np.mean(div):>10.3f}")
    print("\npurity/confidence should rise and diversity fall with w — a tunable knob "
          "scVI does not provide.")

    counterfactual_part(cfg, adata, X, y, classes, clf)


def _cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(1)


def counterfactual_part(cfg, adata, X, y, classes, clf):
    """Part B: SDEdit per-cell counterfactual A->B at varying edit strength."""
    print("\n" + "=" * 70)
    print("COUNTERFACTUAL CELL-STATE EDITING (SDEdit) — per-cell, structure-preserving")
    print("=" * 70)
    dev = get_device(cfg["device"])
    model = load_model("runs/outputs/checkpoints/ddpm_best.pt", dev)
    diff = GaussianDiffusion(
        timesteps=cfg["diffusion"]["timesteps"], beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"], schedule=cfg["diffusion"]["schedule"],
        pred_type=cfg["diffusion"].get("pred_type", "x0"), device=dev)
    gate = cfg["sample"].get("zero_gate", "hurdle")
    calib = cfg["sample"].get("magnitude_calibration", "none") == "quantile"
    mq = getattr(model, "_mag_quant", None)
    name2idx = {c: i for i, c in enumerate(classes)}

    # (A, B): close pair, far pair, and a same-class reconstruction control.
    pairs = [("CD4 T cells", "CD4 T cells"),    # A->A reconstruction (preservation sanity)
             ("CD4 T cells", "CD8 T cells"),    # close
             ("CD14+ Monocytes", "NK cells")]   # far
    strengths = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    rng = np.random.default_rng(0)

    def edit(srcX, b, s):
        return counterfactual_edit(diff, model, srcX, b, s, cfg["sample"]["ddim_steps"],
                                   1.0, getattr(model, "_x0_clip", None),
                                   model._scaler_mean, model._scaler_std, gate,
                                   getattr(model, "_zero_rate", None), dev,
                                   mag_quant=(mq[b] if (calib and mq is not None) else None))

    for A, B in pairs:
        a, b = name2idx[A], name2idx[B]
        n = min((y == a).sum(), (y == b).sum(), 150)
        srcX = X[y == a][rng.choice((y == a).sum(), n, replace=False)]
        freshB = X[y == b][rng.choice((y == b).sum(), n, replace=False)]
        floor = _cos(freshB, srcX).mean()
        tag = "  [A->A reconstruction]" if A == B else ""
        print(f"\n{A}  ->  {B}{tag}   (n={n})")
        print(f"{'strength':>10} {'P(srcA)':>8} {'P(tgtB)':>8} {'->B%':>6} {'preserve':>9}")
        print(f"{'0 (orig)':>10} {clf.predict_proba(srcX)[:, a].mean():>8.3f} "
              f"{clf.predict_proba(srcX)[:, b].mean():>8.3f} {'-':>6} {1.0:>9.3f}")
        for s in strengths:
            ed = edit(srcX, b, s)
            proba = clf.predict_proba(ed)
            print(f"{s:>10.2f} {proba[:, a].mean():>8.3f} {proba[:, b].mean():>8.3f} "
                  f"{(proba.argmax(1) == b).mean():>6.0%} {_cos(ed, srcX).mean():>9.3f}")
        print(f"  (preserve floor = unrelated {B} vs source = {floor:.3f})")

    # Fidelity vs editability tension: the gate+calibration that match distributions
    # also erase per-cell structure. Raw edit (no gate/calib) preserves it.
    print("\n--- Fidelity vs editability (CD4 T -> CD8 T, strength 0.1) ---")
    a, b = name2idx["CD4 T cells"], name2idx["CD8 T cells"]
    n = min((y == a).sum(), (y == b).sum(), 150)
    srcX = X[y == a][rng.choice((y == a).sum(), n, replace=False)]
    floor = _cos(X[y == b][rng.choice((y == b).sum(), n, replace=False)], srcX).mean()
    gated = edit(srcX, b, 0.1)
    raw = counterfactual_edit(diff, model, srcX, b, 0.1, cfg["sample"]["ddim_steps"], 1.0,
                              getattr(model, "_x0_clip", None), model._scaler_mean,
                              model._scaler_std, "none", None, dev, mag_quant=None)
    for nm, ed in [("full pipeline (gate+calib)", gated), ("raw diffusion edit", raw)]:
        pr = clf.predict_proba(ed)
        print(f"  {nm:>28}: ->CD8 {(pr.argmax(1)==b).mean():>4.0%}  "
              f"preserve {_cos(ed, srcX).mean():.3f}  (floor {floor:.3f})")


if __name__ == "__main__":
    main()
