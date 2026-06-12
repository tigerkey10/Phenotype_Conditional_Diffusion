"""Evaluation metrics.

- Distributional fidelity: MMD (multi-bandwidth RBF) and 2-Wasserstein.
  The 2-Wasserstein is computed via the closed-form Gaussian (Frechet) distance,
  W2^2 = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2 (C_r C_g)^{1/2}), evaluated in both
  ambient and PCA-reduced space (same estimator used by FID).
- Biological validity: per-marker KS two-sample test (real vs generated).
- Geometric structure: joint UMAP of real + generated cells (figure).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import linalg
from scipy.stats import ks_2samp, wasserstein_distance


# ----------------------------- MMD -----------------------------------------
def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = (a * a).sum(1)[:, None]
    bb = (b * b).sum(1)[None, :]
    return np.clip(aa + bb - 2.0 * a @ b.T, 0.0, None)


def mmd_rbf(x: np.ndarray, y: np.ndarray, bandwidths: Optional[List[float]] = None) -> float:
    """Unbiased multi-bandwidth RBF-kernel MMD^2 between samples x and y."""
    xx = _pairwise_sq_dists(x, x)
    yy = _pairwise_sq_dists(y, y)
    xy = _pairwise_sq_dists(x, y)

    if bandwidths is None:
        med = np.median(_pairwise_sq_dists(x, y))
        med = med if med > 0 else 1.0
        bandwidths = [med * s for s in (0.25, 0.5, 1.0, 2.0, 4.0)]

    m, n = x.shape[0], y.shape[0]
    total = 0.0
    for bw in bandwidths:
        g = 1.0 / (2.0 * bw)
        kxx = np.exp(-g * xx)
        kyy = np.exp(-g * yy)
        kxy = np.exp(-g * xy)
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)
        term_xx = kxx.sum() / (m * (m - 1)) if m > 1 else 0.0
        term_yy = kyy.sum() / (n * (n - 1)) if n > 1 else 0.0
        total += term_xx + term_yy - 2.0 * kxy.mean()
    return float(total / len(bandwidths))


# --------------------------- Wasserstein-2 ----------------------------------
def wasserstein2_gaussian(x: np.ndarray, y: np.ndarray) -> float:
    """Closed-form 2-Wasserstein between Gaussian approximations (FID formula)."""
    mu1, mu2 = x.mean(0), y.mean(0)
    c1 = np.cov(x, rowvar=False)
    c2 = np.cov(y, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = diff @ diff + np.trace(c1 + c2 - 2.0 * covmean)
    return float(np.sqrt(max(dist, 0.0)))


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int = 200, seed: int = 0) -> float:
    """Sliced 2-Wasserstein: mean of 1D W2 over random projections."""
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    dirs = rng.standard_normal((n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    vals = []
    for v in dirs:
        vals.append(wasserstein_distance(x @ v, y @ v))
    return float(np.mean(vals))


# ----------------------------- markers --------------------------------------
def marker_ks(real: np.ndarray, gen: np.ndarray) -> Dict[str, float]:
    stat, p = ks_2samp(real, gen)
    return {"ks_stat": float(stat), "ks_p": float(p),
            "real_mean": float(real.mean()), "gen_mean": float(gen.mean())}


# --------------------------- aggregate driver -------------------------------
def distributional_metrics(
    real: np.ndarray,
    gen: np.ndarray,
    pca_real: Optional[np.ndarray] = None,
    pca_gen: Optional[np.ndarray] = None,
    subsample: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    def _sub(a):
        if a.shape[0] > subsample:
            return a[rng.choice(a.shape[0], subsample, replace=False)]
        return a

    r, g = _sub(real), _sub(gen)
    # Ambient 2-Wasserstein is estimated with the sliced estimator: the
    # closed-form Gaussian W2 needs an O(G^3) matrix square root on a
    # rank-deficient G x G (G=1000) covariance, which is both ill-conditioned
    # and ~80s per call. The Gaussian W2 is reserved for the full-rank PCA space.
    out = {
        "mmd_ambient": mmd_rbf(r, g),
        "w2_ambient": sliced_wasserstein(r, g, seed=seed),
    }
    if pca_real is not None and pca_gen is not None:
        pr, pg = _sub(pca_real), _sub(pca_gen)
        out["mmd_pca"] = mmd_rbf(pr, pg)
        out["w2_pca_gauss"] = wasserstein2_gaussian(pca_real, pca_gen)
        out["w2_pca_sliced"] = sliced_wasserstein(pr, pg, seed=seed)
    return out
