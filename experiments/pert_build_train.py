"""Kang IFN-beta perturbation benchmark — build data + train the conditional diffusion.

Task (scGen benchmark): hold out one cell type's STIMULATED cells. Train a diffusion
conditioned on condition (ctrl/stim) on everything else (all ctrl + other types' stim).
At eval, predict the held-out type's stim from its ctrl by counterfactual editing.

This script builds the preprocessed cache and trains the model into runs/outputs_kang/.
Run:  python experiments/pert_build_train.py [holdout_celltype] [epochs]
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import anndata as ad
import scanpy as sc
import yaml

from src.utils import set_seed, get_device

HOLDOUT = sys.argv[1] if len(sys.argv) > 1 else "CD4 T cells"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
N_HVG = 2000
CACHE = "runs/outputs_kang/kang_hvg.h5ad"


def build_cache():
    if os.path.exists(CACHE):
        return ad.read_h5ad(CACHE)
    os.makedirs("runs/outputs_kang", exist_ok=True)
    a = ad.read_h5ad("data/kang_2018.h5ad")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG)
    a = a[:, a.var.highly_variable].copy()
    a.X = np.asarray(a.X.todense()) if hasattr(a.X, "todense") else np.asarray(a.X)
    a.X = a.X.astype(np.float32)
    a.obs["label"] = (a.obs["label"].astype(str) == "stim").astype(int)  # 0 ctrl, 1 stim
    a.uns["classes"] = np.array(["ctrl", "stim"], dtype=object)
    a.write_h5ad(CACHE)
    return a


def main():
    from src.train import train
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["out_dir"] = "runs/outputs_kang"
    cfg["data"]["cache"] = CACHE
    cfg["data"]["n_hvg"] = N_HVG
    cfg["train"]["epochs"] = EPOCHS
    cfg["train"]["eval_every"] = 25
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    adata = build_cache()
    y = adata.obs["label"].values.astype(int)
    ct = adata.obs["cell_type"].astype(str).values
    print(f"data: {adata.shape}, holdout = {HOLDOUT} stim")

    # training pool = everything EXCEPT the held-out cell type's stim cells
    holdout_stim = (ct == HOLDOUT) & (y == 1)
    pool = np.where(~holdout_stim)[0]
    print(f"held-out {HOLDOUT} stim cells excluded: {holdout_stim.sum()}; train pool: {len(pool)}")

    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(pool)
    n_val = int(0.1 * len(pool))
    val_idx = np.sort(pool[:n_val])
    train_idx = np.sort(pool[n_val:])

    info = train(cfg, adata, train_idx, val_idx, device)
    print("trained:", info["best"])


if __name__ == "__main__":
    main()
