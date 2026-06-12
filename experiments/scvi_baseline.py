"""scVI baseline — a real domain SOTA generative model (Lopez et al., 2018).

Trains scVI on the same cells (raw counts, full filtered gene set, train split only),
then generates per-class synthetic cells by posterior-predictive sampling on
bootstrap-resampled real cells of each type. The generated counts are pushed through
the *identical* preprocessing as the real data (CP10K -> log1p -> subset to the saved
1000 HVGs), so scVI samples land in exactly the evaluation space used for v1/v2.

Outputs: runs/outputs_scvi/samples/scvi.npz (keys class_0..class_7) and distributional
metrics vs the real test split.

Run:  python experiments/scvi_baseline.py [epochs]
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
import scvi
import yaml

from src import data as datamod
from src.evaluate import distributional_metrics
from sklearn.decomposition import PCA


def build_raw_counts(cfg):
    """Raw counts for the same cells / filtered genes as preprocessing."""
    proc = sc.datasets.pbmc3k_processed()
    labels = proc.obs["louvain"].astype(str)
    raw = sc.datasets.pbmc3k()
    raw.var_names_make_unique()
    sc.pp.filter_cells(raw, min_genes=cfg["data"]["min_genes"])
    sc.pp.filter_genes(raw, min_cells=cfg["data"]["min_cells"])
    shared = raw.obs_names.intersection(proc.obs_names)
    raw = raw[shared].copy()
    raw.obs["cell_type"] = labels.loc[shared].values
    raw.X = np.asarray(raw.X.todense()) if hasattr(raw.X, "todense") else np.asarray(raw.X)
    return raw


def to_eval_space(counts, var_names, hvg_names, target_sum):
    """Apply the real-data preprocessing to generated counts -> log-norm HVG."""
    a = ad.AnnData(np.asarray(counts, np.float32))
    a.var_names = var_names
    sc.pp.normalize_total(a, target_sum=target_sum)
    sc.pp.log1p(a)
    a = a[:, hvg_names].copy()
    return np.asarray(a.X, np.float32)


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    cfg = yaml.safe_load(open("configs/default.yaml"))
    scvi.settings.seed = cfg["seed"]

    # eval-space reference (our X) + split + HVG names from the cache
    adata = datamod.preprocess(cfg)
    hvg_names = list(adata.var_names)
    classes = datamod.classes_of(adata)
    n_classes = len(classes)
    y = adata.obs["label"].values.astype(int)
    tr, va, te = datamod.split_indices(len(y), cfg["data"]["val_frac"],
                                       cfg["data"]["test_frac"], y, cfg["seed"])
    test_X = np.asarray(adata.X, np.float32)[te]
    test_y = y[te]
    train_X = np.asarray(adata.X, np.float32)[tr]

    # raw counts, aligned to the cache cell order, train split only for scVI
    raw = build_raw_counts(cfg)
    raw = raw[adata.obs_names].copy()           # same order as cache
    raw_train = raw[tr].copy()
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    raw_train.obs["label"] = raw_train.obs["cell_type"].map(cls_to_idx).astype(int)

    raw_train.layers["counts"] = raw_train.X.copy()
    scvi.model.SCVI.setup_anndata(raw_train, layer="counts")
    model = scvi.model.SCVI(raw_train, n_latent=10)
    model.train(max_epochs=epochs, early_stopping=False,
                plan_kwargs={"lr": 1e-3}, accelerator="auto")

    # generate M per class by posterior-predictive on bootstrapped class cells
    M = cfg["sample"]["n_per_class"]
    rng = np.random.default_rng(cfg["seed"])
    labels_train = raw_train.obs["label"].values
    var_names = list(raw_train.var_names)
    samples = {}
    for c in range(n_classes):
        ci = np.where(labels_train == c)[0]
        boot = rng.choice(ci, M, replace=True)
        pp = model.posterior_predictive_sample(raw_train, indices=boot, n_samples=1)
        counts = np.asarray(pp.todense()) if hasattr(pp, "todense") else np.asarray(pp)
        counts = counts.reshape(len(boot), -1)
        samples[c] = to_eval_space(counts, var_names, hvg_names,
                                   cfg["data"]["target_sum"])
        print(f"  class {c} ({classes[c]}): generated {samples[c].shape}", flush=True)

    os.makedirs("runs/outputs_scvi/samples", exist_ok=True)
    np.savez_compressed("runs/outputs_scvi/samples/scvi.npz",
                        **{f"class_{c}": v for c, v in samples.items()})
    print("saved runs/outputs_scvi/samples/scvi.npz")

    # distributional metrics vs test set (same pipeline as main eval)
    def pca_fit(trX, *arrs, dim):
        p = PCA(n_components=dim, random_state=0).fit(trX)
        return [p.transform(a) for a in arrs]

    rows = []
    for c in range(n_classes):
        real = test_X[test_y == c]
        if real.shape[0] < 2:
            continue
        pr, pg = pca_fit(train_X, real, samples[c], dim=cfg["eval"]["pca_dim"])
        rows.append(distributional_metrics(real, samples[c], pr, pg,
                    subsample=cfg["eval"]["mmd_subsample"], seed=cfg["seed"]))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    g = np.concatenate([samples[c] for c in range(n_classes)], 0)
    print("\n=== scVI distributional metrics (class-avg) ===")
    print("zero%% %.1f  nonzero_mean %.3f"%((g == 0).mean()*100, g[g > 0].mean()))
    for k in ["mmd_ambient", "mmd_pca", "w2_ambient", "w2_pca_gauss"]:
        print(f"  {k:16s} {agg[k]:.4f}")


if __name__ == "__main__":
    main()
