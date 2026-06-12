"""End-to-end driver for Phenotype-Conditional Diffusion on PBMC 3k.

Stages:
  preprocess  build the HVG-reduced, labeled AnnData cache
  train       train the 1D conditional DDPM (best-val-MMD checkpoint)
  sample      draw N cells/class at each guidance scale
  evaluate    DDPM + baselines: distributional metrics, marker KS, joint UMAP
  all         run every stage in order

Usage:
  python main.py all --config configs/default.yaml
  python main.py train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np

from src import data as datamod
from src.utils import load_config, set_seed, get_device, ensure_dir


# ---------------------------------------------------------------------------
def stage_preprocess(cfg) -> None:
    adata = datamod.preprocess(cfg, force=False)
    print(f"preprocessed: X {adata.X.shape}, classes={list(adata.uns['classes'])}")


def _splits(cfg, adata):
    y = adata.obs["label"].values.astype(np.int64)
    return datamod.split_indices(
        len(y), cfg["data"]["val_frac"], cfg["data"]["test_frac"], y, cfg["seed"]
    )


def stage_train(cfg, device) -> Dict[str, str]:
    import torch  # noqa
    from src.train import train
    adata = datamod.preprocess(cfg)
    tr, va, te = _splits(cfg, adata)
    print(f"split: train {len(tr)}  val {len(va)}  test {len(te)}")
    return train(cfg, adata, tr, va, device)


def stage_sample(cfg, device) -> None:
    from src.sample import load_model, sample_all_classes, save_samples
    ckpt = os.path.join(cfg["out_dir"], "checkpoints", "ddpm_best.pt")
    model = load_model(ckpt, device)
    for w in cfg["sample"]["guidance_scales"]:
        samples = sample_all_classes(cfg, model, device, w=float(w))
        path = save_samples(cfg, samples, w=float(w))
        print(f"sampled w={w}: {path}")


# ---------------------------------------------------------------------------
def _pca_fit_transform(train_X, *arrays, dim):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim, random_state=0).fit(train_X)
    return [pca.transform(a) for a in arrays]


def stage_evaluate(cfg, device) -> None:
    import torch
    from src.evaluate import distributional_metrics, marker_ks
    from src.sample import load_model, sample_all_classes
    from src.baselines import ConditionalVAE, train_cvae, sample_cvae, PerClassGaussian

    adata = datamod.preprocess(cfg)
    classes = datamod.classes_of(adata)
    X = np.asarray(adata.X, dtype=np.float32)
    y = adata.obs["label"].values.astype(np.int64)
    n_classes = len(classes)
    tr, va, te = _splits(cfg, adata)
    train_X, train_y = X[tr], y[tr]
    test_X, test_y = X[te], y[te]

    fig_dir = ensure_dir(os.path.join(cfg["out_dir"], "figures"))
    metrics_dir = ensure_dir(os.path.join(cfg["out_dir"], "metrics"))

    # ---- DDPM samples (best guidance scale + all scales recorded) ----------
    model = load_model(os.path.join(cfg["out_dir"], "checkpoints", "ddpm_best.pt"), device)
    ddpm_by_w = {
        float(w): sample_all_classes(cfg, model, device, w=float(w))
        for w in cfg["sample"]["guidance_scales"]
    }

    # ---- Baselines ---------------------------------------------------------
    vae = ConditionalVAE(X.shape[1], n_classes,
                         latent_dim=cfg["baseline"]["vae"]["latent_dim"],
                         hidden=cfg["baseline"]["vae"]["hidden"])
    vae = train_cvae(vae, train_X, train_y, cfg["baseline"]["vae"], device)
    vae_samples = {c: sample_cvae(vae, cfg["sample"]["n_per_class"], c, device)
                   for c in range(n_classes)}

    pcg = PerClassGaussian(pca_dim=cfg["eval"]["pca_dim"], seed=cfg["seed"]).fit(train_X, train_y)
    gauss_samples = {c: pcg.sample(cfg["sample"]["n_per_class"], c) for c in range(n_classes)}

    # ---- Distributional metrics per class & method -------------------------
    def eval_method(name, samples_by_class):
        rows = {}
        for c in range(n_classes):
            real = test_X[test_y == c]
            gen = samples_by_class[c]
            if real.shape[0] < 2:
                continue
            pr, pg = _pca_fit_transform(train_X, real, gen, dim=cfg["eval"]["pca_dim"])
            m = distributional_metrics(
                real, gen, pca_real=pr, pca_gen=pg,
                subsample=cfg["eval"]["mmd_subsample"], seed=cfg["seed"],
            )
            rows[classes[c]] = m
        # aggregate (macro-average over classes)
        agg = {}
        if rows:
            for k in next(iter(rows.values())):
                agg[k] = float(np.mean([r[k] for r in rows.values()]))
        return {"per_class": rows, "aggregate": agg}

    results = {"classes": classes, "methods": {}}
    for w, samp in ddpm_by_w.items():
        results["methods"][f"ddpm_w{w}"] = eval_method(f"ddpm_w{w}", samp)
    results["methods"]["cvae"] = eval_method("cvae", vae_samples)
    results["methods"]["gaussian_pca"] = eval_method("gaussian_pca", gauss_samples)

    # ---- Marker-gene KS test (best DDPM scale w=1 by convention) ------------
    best_w = 1.0 if 1.0 in ddpm_by_w else float(cfg["sample"]["guidance_scales"][0])
    markers = list(cfg["eval"]["markers"].keys())
    hvg_names = list(adata.var_names)
    marker_results = {}
    for g in markers:
        if g not in hvg_names:
            continue  # marker not among HVGs; skip (still reported as absent)
        gi = hvg_names.index(g)
        per_class = {}
        for c in range(n_classes):
            real = test_X[test_y == c][:, gi]
            gen = ddpm_by_w[best_w][c][:, gi]
            if real.shape[0] >= 2:
                per_class[classes[c]] = marker_ks(real, gen)
        marker_results[g] = {"expected_high_in": cfg["eval"]["markers"][g],
                             "per_class": per_class, "in_hvg": True}
    for g in markers:
        if g not in hvg_names:
            marker_results[g] = {"expected_high_in": cfg["eval"]["markers"][g], "in_hvg": False}

    results["markers"] = {"guidance_scale": best_w, "results": marker_results}

    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_csv(results, os.path.join(metrics_dir, "summary.csv"))
    print(f"metrics -> {metrics_dir}/metrics.json")
    _print_summary(results)

    # ---- Joint UMAP --------------------------------------------------------
    _umap_figure(cfg, test_X, test_y, ddpm_by_w[best_w], classes,
                 os.path.join(fig_dir, "umap_real_vs_ddpm.png"), best_w)


def _write_csv(results, path):
    import csv
    keys = ["mmd_ambient", "mmd_pca", "w2_ambient", "w2_pca_gauss", "w2_pca_sliced"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + keys)
        for m, res in results["methods"].items():
            agg = res["aggregate"]
            w.writerow([m] + [f"{agg.get(k, float('nan')):.5f}" for k in keys])


def _print_summary(results):
    print("\n=== Aggregate (macro-avg over classes) ===")
    header = f"{'method':<14} {'mmd_amb':>9} {'mmd_pca':>9} {'w2_amb':>9} {'w2_pca':>9}"
    print(header)
    for m, res in results["methods"].items():
        a = res["aggregate"]
        print(f"{m:<14} {a.get('mmd_ambient', float('nan')):9.4f} "
              f"{a.get('mmd_pca', float('nan')):9.4f} "
              f"{a.get('w2_ambient', float('nan')):9.4f} "
              f"{a.get('w2_pca_gauss', float('nan')):9.4f}")


def _umap_figure(cfg, real_X, real_y, gen_by_class, classes, path, w):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    gen_X = np.concatenate([gen_by_class[c] for c in range(len(classes))], axis=0)
    gen_y = np.concatenate([np.full(gen_by_class[c].shape[0], c) for c in range(len(classes))])

    joint = np.concatenate([real_X, gen_X], axis=0)
    origin = np.array(["real"] * real_X.shape[0] + ["gen"] * gen_X.shape[0])
    lbl = np.concatenate([real_y, gen_y])

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=cfg["seed"])
    emb = reducer.fit_transform(joint)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.get_cmap("tab10")
    for c in range(len(classes)):
        m = lbl == c
        axes[0].scatter(emb[m, 0], emb[m, 1], s=6, color=cmap(c % 10), label=classes[c], alpha=0.6)
    axes[0].set_title("Joint UMAP — colored by cell type")
    axes[0].legend(fontsize=7, markerscale=2, loc="best")

    for o, col, mk in [("real", "tab:blue", "o"), ("gen", "tab:red", "x")]:
        m = origin == o
        axes[1].scatter(emb[m, 0], emb[m, 1], s=6, c=col, marker=mk, label=o, alpha=0.5)
    axes[1].set_title(f"Joint UMAP — real vs generated (DDPM w={w})")
    axes[1].legend(markerscale=2)

    for ax in axes:
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"figure -> {path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preprocess", "train", "sample", "evaluate", "all"])
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dir(cfg["out_dir"])
    device = get_device(cfg["device"])
    print(f"device: {device}")

    if args.stage in ("preprocess", "all"):
        stage_preprocess(cfg)
    if args.stage in ("train", "all"):
        stage_train(cfg, device)
    if args.stage in ("sample", "all"):
        stage_sample(cfg, device)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(cfg, device)


if __name__ == "__main__":
    main()
