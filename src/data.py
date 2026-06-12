"""PBMC 3k loading and preprocessing.

Standard scanpy pipeline: filter -> normalize_total(1e4) -> log1p -> top-G HVG.
Cell-type labels (8 classes) are taken from `pbmc3k_processed()` (louvain
annotation) and matched to the filtered raw matrix by cell barcode, yielding the
final input matrix X in R^{2638 x G} with per-cell integer label c in {0..7}.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import scanpy as sc


def _to_dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


def preprocess(cfg: dict, force: bool = False) -> ad.AnnData:
    """Build (or load cached) the preprocessed, HVG-reduced, labeled AnnData."""
    cache = cfg["data"]["cache"]
    if os.path.exists(cache) and not force:
        return ad.read_h5ad(cache)

    os.makedirs(os.path.dirname(cache), exist_ok=True)

    # Labels from the processed (annotated) release.
    proc = sc.datasets.pbmc3k_processed()
    labels = proc.obs["louvain"].astype(str)

    # Raw counts -> standard pipeline.
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=cfg["data"]["min_genes"])
    sc.pp.filter_genes(adata, min_cells=cfg["data"]["min_cells"])

    # Keep only cells that carry an annotation (intersection of barcodes).
    shared = adata.obs_names.intersection(proc.obs_names)
    adata = adata[shared].copy()
    adata.obs["cell_type"] = labels.loc[shared].values

    sc.pp.normalize_total(adata, target_sum=cfg["data"]["target_sum"])
    sc.pp.log1p(adata)

    # Keep a copy of all log-norm genes so marker genes survive HVG selection.
    adata.raw = adata

    n_hvg = cfg["data"]["n_hvg"]
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)

    # Force-include the canonical marker genes used for biological validation:
    # the model only ever generates the selected gene dimensions, so markers
    # named in the evaluation (CD3D, MS4A1, ...) must be in the modeled set.
    # We keep exactly G = n_hvg genes by swapping in any missing markers for the
    # lowest-ranked HVGs.
    markers = list(cfg.get("eval", {}).get("markers", {}).keys())
    present = set(adata.var_names)
    force = [g for g in markers if g in present]
    hv = adata.var["highly_variable"].copy()
    missing = [g for g in force if not bool(hv[g])]
    if missing:
        ranked = adata.var.sort_values("dispersions_norm", ascending=False).index
        selected = [g for g in ranked if bool(hv[g])]
        keep = set(selected[: max(0, n_hvg - len(missing))]) | set(missing)
        hv = adata.var_names.isin(keep)
        adata.var["highly_variable"] = hv
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Integer label encoding (stable, sorted by name).
    classes: List[str] = sorted(adata.obs["cell_type"].unique().tolist())
    class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}
    adata.obs["label"] = adata.obs["cell_type"].map(class_to_idx).astype(int)
    adata.uns["classes"] = np.array(classes, dtype=object)

    # Densify X for downstream torch use.
    adata.X = _to_dense(adata.X).astype(np.float32)
    adata.write_h5ad(cache)
    return adata


def split_indices(
    n: int, val_frac: float, test_frac: float, labels: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split so every class appears in each split."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        n_val = max(1, int(round(len(idx) * val_frac)))
        test.extend(idx[:n_test])
        val.extend(idx[n_test : n_test + n_val])
        train.extend(idx[n_test + n_val :])
    return (
        np.array(sorted(train)),
        np.array(sorted(val)),
        np.array(sorted(test)),
    )


def get_marker_matrix(adata: ad.AnnData, genes: List[str]) -> Dict[str, np.ndarray]:
    """Per-gene log-norm expression vector pulled from the full (raw) gene set."""
    source = adata.raw.to_adata() if adata.raw is not None else adata
    out: Dict[str, np.ndarray] = {}
    for g in genes:
        if g in source.var_names:
            out[g] = _to_dense(source[:, g].X).ravel().astype(np.float32)
    return out


def classes_of(adata: ad.AnnData) -> List[str]:
    return list(adata.uns["classes"])
