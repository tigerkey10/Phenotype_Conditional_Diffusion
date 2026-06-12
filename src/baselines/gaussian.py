"""Per-class Gaussian baseline in PCA-reduced space.

Fits a full-covariance Gaussian per class on PCA coordinates, samples from it,
and inverse-transforms back to the ambient log-norm gene space.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.decomposition import PCA


class PerClassGaussian:
    def __init__(self, pca_dim: int = 50, seed: int = 0):
        self.pca = PCA(n_components=pca_dim, random_state=seed)
        self.means: Dict[int, np.ndarray] = {}
        self.covs: Dict[int, np.ndarray] = {}
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerClassGaussian":
        Z = self.pca.fit_transform(X)
        for c in np.unique(y):
            zc = Z[y == c]
            self.means[int(c)] = zc.mean(0)
            self.covs[int(c)] = np.cov(zc, rowvar=False) + 1e-6 * np.eye(zc.shape[1])
        return self

    def sample(self, n: int, c_idx: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + c_idx)
        z = rng.multivariate_normal(self.means[c_idx], self.covs[c_idx], size=n)
        return self.pca.inverse_transform(z).astype(np.float32)
