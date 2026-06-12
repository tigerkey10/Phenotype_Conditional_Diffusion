"""Downstream augmentation experiment.

Tests the proposal's stated motivation — synthetic cells as data augmentation for
downstream classification, especially of rare cell types. In a data-scarce regime
(K real cells per class) we train a cell-type classifier on:
  - real-only, vs
  - real + M synthetic cells/class from each generator (scVI / v1 / v2),
and evaluate on the held-out REAL test split. We report accuracy and macro-F1
(which weights every class equally, so rare-class gains show up).

Run:  python experiments/downstream_augmentation.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from src import data as datamod

# Generator sample files (npz with keys class_0..class_7). Missing files are skipped.
GENERATORS = {
    "scVI":        "runs/outputs_scvi/samples/scvi.npz",
    "v1 (calib)":  "runs/outputs_v1_calibrated/samples/ddpm_w1.0.npz",
    "v2 (end2end)":"runs/outputs_end2end/samples/ddpm_w1.0.npz",
}
K_LIST = [10, 25, 50]      # real cells per class (data-scarce regimes)
M_AUG = 200                # synthetic cells per class added
SEED = 0


def few_shot_subsample(X, y, k, seed):
    """Keep at most k real cells per class — the data-scarce regime augmentation helps."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        idx.extend(ci[:k])          # classes with < k cells contribute all they have
    idx = np.array(idx)
    return X[idx], y[idx]


def load_gen(path, n_classes):
    """Load a generator's per-class samples: npz with keys class_0..class_{n-1}."""
    d = np.load(path)
    return {c: d[f"class_{c}"] for c in range(n_classes)}


def augment(Xr, yr, gen, m, n_classes, seed):
    """Append m synthetic cells per class to the few-shot real set (Xr, yr)."""
    rng = np.random.default_rng(seed + 1)
    Xs, ys = [Xr], [yr]
    for c in range(n_classes):
        a = gen[c]
        sel = rng.choice(a.shape[0], min(m, a.shape[0]), replace=False)
        Xs.append(a[sel]); ys.append(np.full(len(sel), c))
    return np.concatenate(Xs), np.concatenate(ys)


def train_eval(Xtr, ytr, Xte, yte):
    """Fit a logistic-regression cell-type classifier; return (accuracy, macro-F1)
    on the real test set. macro-F1 weights every class equally, so it surfaces gains
    on rare classes that overall accuracy would hide."""
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")


def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    adata = datamod.preprocess(cfg)
    X = np.asarray(adata.X, np.float32)
    y = adata.obs["label"].values.astype(int)
    classes = datamod.classes_of(adata)
    n_classes = len(classes)
    tr, va, te = datamod.split_indices(len(y), cfg["data"]["val_frac"],
                                       cfg["data"]["test_frac"], y, cfg["seed"])
    Xtr_all, ytr_all = X[tr], y[tr]    # real training pool to subsample from
    Xte, yte = X[te], y[te]            # real held-out test set (never augmented)

    # load whichever generator sample files exist (scVI / v1 / v2)
    gens = {name: load_gen(p, n_classes) for name, p in GENERATORS.items()
            if os.path.exists(p)}
    print(f"generators available: {list(gens.keys())}")
    print(f"test set: {len(yte)} real cells\n")

    header = f"{'K/class':>8} {'real-only':>11} " + " ".join(
        f"{('+'+g):>16}" for g in gens)
    print("ACCURACY / MACRO-F1 on real test set")
    print(header)
    for k in K_LIST:                                   # sweep scarcity (real cells/class)
        Xr, yr = few_shot_subsample(Xtr_all, ytr_all, k, SEED)
        acc0, f10 = train_eval(Xr, yr, Xte, yte)       # real-only baseline classifier
        row = f"{k:>8} {acc0:.3f}/{f10:.3f}  "
        for name, gen in gens.items():                 # then real + each generator's synthetics
            Xa, ya = augment(Xr, yr, gen, M_AUG, n_classes, SEED)
            acc, f1 = train_eval(Xa, ya, Xte, yte)
            d = f1 - f10
            row += f"  {acc:.3f}/{f1:.3f}({d:+.3f})"
        print(row)
    print("\n(value = accuracy/macro-F1; (Δ) = macro-F1 gain over real-only)")


if __name__ == "__main__":
    main()
