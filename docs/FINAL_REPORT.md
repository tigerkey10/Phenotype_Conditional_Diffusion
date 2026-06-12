# Final Report — Phenotype-Conditional Diffusion for Single-Cell Gene Expression Generation

> Draft material for the final report. All numbers are taken directly from the
> submitted code and runs (see `README.md` §1–13 and `DATA.md`). Honest negative
> results are reported alongside the positive ones.

---

## 1. Project Title and Team Member Information

- **Project title:** Phenotype-Conditional Diffusion for Single-Cell Gene Expression Generation
- **Team member:** Kibeom Kim — Student ID **202490517** (single-member project)

---

## 2. Brief Description of the Research Project

This project builds a **conditional denoising-diffusion model (DDPM) with classifier-free
guidance (CFG)** that generates single-cell RNA-seq (scRNA-seq) gene-expression profiles
conditioned on a phenotype label (cell type, or experimental condition). The aim is to
transfer two ideas that define the state of the art in *vision* generative modeling —
**denoising diffusion** and **classifier-free guidance** — into a non-image biological
modality, and to test, under a controlled and fully reproducible setup, how faithfully they
transfer.

The work was carried out in three expanding stages:

1. **Core transfer (as proposed):** a minimal 1D conditional DDPM on the public **PBMC-3k**
   dataset (2,638 cells × 1,000 highly variable genes, 8 annotated cell types), generating a
   synthetic log-normalized expression vector per target cell type.
2. **Making it actually work:** a literal implementation of the proposal produced broken
   samples; the project diagnosed *why* and added five standard, documented fixes —
   culminating in a model whose generated cells are statistically and biologically faithful
   and that **beats simple baselines (conditional VAE, per-class Gaussian)**.
3. **Stress-testing for a real contribution:** the model was then benchmarked against a
   genuine domain SOTA (**scVI**), used for **downstream data augmentation**, probed for
   **controllable / counterfactual generation**, and finally moved to a real **perturbation-
   prediction benchmark (Kang et al. 2018 IFN-β)** against **scGen**. These experiments
   answer the question "does this approach beat the field?" — and the honest answer is *not
   yet*, which the report documents rigorously.

---

## 3. Motivation & Problem Statement

**Why the problem matters.** scRNA-seq experiments are expensive, sparse, and frequently lack
samples of rare cell types or rare perturbation responses. High-fidelity *conditional*
generators can serve as data augmentation for downstream classification, recover rare
populations, and predict unseen perturbation states *in silico*. Historically, biology-facing
generative work on scRNA-seq has relied on **VAE-based** models (scVI, scVAE), optimized for
representation learning and clustering rather than high-fidelity conditional synthesis.
Meanwhile, **diffusion models** have overtaken VAEs/GANs for conditional generation of
structured data in vision. Whether that advantage transfers to the discrete-ish, zero-inflated
world of gene expression is an open and practically relevant question.

**Main challenge / research question.** The central question of the proposal was:

> *Can a minimal 1D DDPM with classifier-free guidance, trained directly on log-normalized
> expression vectors, generate cell-type-conditional scRNA-seq samples that are statistically
> and biologically faithful to the reference distribution?*

In the course of answering it, two harder, more honest questions emerged and were pursued:

- **(Engineering)** *Why* does a vision-style DDPM break on scRNA-seq, and what is the minimal
  set of modality-specific fixes that makes it work? (Answer: scale, sampling stability, and
  **zero-inflation** — the 93%-zero structure of the data.)
- **(Scientific)** Once it works, does the diffusion approach offer any *demonstrable advantage*
  over established single-cell generative models (scVI, scGen) on a real task?

---

## 4. Proposed Method / Technical Details

All components below correspond directly to the submitted code (`src/`, `experiments/`,
`configs/`); file references are given so the description stays consistent with the artifacts.

### 4.1 Data and preprocessing (`src/data.py`, `experiments/pert_build_train.py`)

- **PBMC-3k** (`scanpy.datasets.pbmc3k`): raw counts → filter cells (≥200 genes) and genes
  (≥3 cells) → `normalize_total(1e4)` → `log1p` → top **G = 1,000** highly variable genes.
  Cell-type labels (8 classes) come from `pbmc3k_processed()` (Louvain), matched by barcode.
  Final input matrix **X ∈ ℝ^{2638×1000}**, integer label `c ∈ {0..7}`. The 8 canonical marker
  genes (CD3D, MS4A1, LYZ, NKG7, GNLY, FCGR3A, PPBP, CD8A) are force-included into the HVG set
  so they exist in the modeled space. Stratified 80/10/10 train/val/test split.
- **Kang et al. 2018** (`data/kang_2018.h5ad`, 24,673 cells × 15,706 genes): PBMC, control vs
  IFN-β-stimulated, 8 cell types. Same preprocessing → G = 2,000 HVG; label = condition
  (0 = ctrl, 1 = stim).

### 4.2 Generative model (`src/model.py`, `src/diffusion.py`)

- **Denoiser** `f_θ(x_t, t, c)`: a residual-MLP backbone — 4 residual blocks, hidden width 512,
  SiLU + LayerNorm — with a sinusoidal time embedding and a learned class embedding added at
  every block. CFG uses a dedicated null-class index.
- **Forward process** (variance preserving): `q(x_t|x_0) = N(√ᾱ_t·x_0, (1−ᾱ_t)·I)`, T = 1000
  steps, **cosine** β-schedule (linear available).
- **Training objective:** the network predicts the clean signal **x₀** (eps-prediction
  available via `pred_type`). Loss is MSE; CFG drops the label to the null token with p = 0.1.
- **Sampling:** deterministic **DDIM**, 50 steps, with per-gene x₀ clipping for stability.
- **SDEdit editing** (`ddim_edit`): noise a real cell partway down the trajectory, then
  denoise toward a *different* target class — used for counterfactual / perturbation editing.

### 4.3 Five design choices that make diffusion work on scRNA-seq (`README.md` §5)

A literal eps-prediction DDPM on log-norm vectors produces generated values ~200× the real
scale. Diagnosis revealed three failure modes; five fixes (all configurable) resolve them:

| # | Fix | Why it is needed |
|---|-----|------------------|
| 1 | **Per-gene z-score standardization** | log-norm data has tiny per-gene variance + 93% zeros, so the unit-variance forward noise drowns the signal. |
| 2 | **Per-gene x₀ clipping in DDIM** | x₀ is divided by √ᾱ_t ≈ 6e-3 at the first step, amplifying the irreducible noise-prediction error ~150× → divergence. |
| 3 | **x₀-prediction + cosine schedule** | eps-prediction leaves a large positive mean bias on this sparse data; predicting x₀ removes it. |
| 4 | **Learned hurdle gate** (2nd denoiser head, BCE) | continuous diffusion cannot emit exact zeros; a learned per-cell dropout gate restores the ~93% sparsity. |
| 5 | **Quantile magnitude calibration** | the diffusion compresses dynamic range; rank-preserving mapping onto real per-gene quantiles restores expressed-gene magnitude. |

A **diagnostic progression** (generated mean / std at each stage) documents the repair:
vanilla eps (mean −0.05, std 123, exploded) → +standardization (std 50) → +x₀-clip
(mean 2.33) → +x₀-prediction (mean 0.29) → +gate+calibration (mean 0.15, matching real 0.13).

### 4.4 Two model variants

- **v1 (`configs/default.yaml`):** all-cell standardization + learned gate + quantile
  calibration. Lowest distributional distance, but calibration *injects* real per-gene marginals.
- **v2 (`configs/end2end.yaml`):** **expressed-only** standardization (per-gene mean/std over
  *nonzero* cells), **no** calibration. Magnitudes are produced entirely by the generative model
  (only 2 stats/gene used) — the more defensible, fully-generative variant.

### 4.5 Baselines and downstream tasks (`src/baselines/`, `experiments/`)

- **cVAE** and **per-class Gaussian (PCA)** — confirm diffusion beats trivial latent-Gaussian.
- **scVI** (`experiments/scvi_baseline.py`) — real domain SOTA (NB likelihood), generates by
  posterior-predictive sampling, pushed through identical preprocessing.
- **Downstream augmentation** (`experiments/downstream_augmentation.py`) — few-shot cell-type
  classification, augmenting K real cells/class with synthetic cells.
- **Controllable generation** (`experiments/controllable_generation.py`) — CFG-strength control
  and per-cell SDEdit counterfactual editing.
- **Perturbation prediction** (`experiments/pert_*.py`) — Kang IFN-β scGen benchmark: hold out
  one cell type's *stimulated* cells; predict them from that type's control cells, having learned
  the ctrl→stim shift only from the *other* types. Diffusion = SDEdit toward "stim" + hurdle
  gate; scGen = latent vector arithmetic.

### 4.6 Evaluation metrics (`src/evaluate.py`)

- **Distributional fidelity:** multi-bandwidth RBF **MMD** and **2-Wasserstein** in ambient and
  PCA-50 space (Gaussian/Fréchet and sliced estimators).
- **Biological validity:** per-marker two-sample **KS test** (real vs generated).
- **Geometric structure:** joint **UMAP** of real + generated cells.
- **Perturbation:** R² of mean expression on top-50 DEGs, R² of the *change* (stim−ctrl) on
  DEGs, per-DEG 1-Wasserstein, variance match.

**Compute:** PyTorch + scanpy + scvi-tools. The full PBMC pipeline trains in ~2–3 min on one
modern GPU (the proposal budgeted <2 h on an RTX 3090). Single seed (`seed: 0`).

---

## 5. Results & Discussion

### 5.1 Core result — the diffusion works and beats simple baselines (PBMC-3k)

After the five fixes, generated cells match the real **sparsity (92.2% vs 93.2% zeros)**,
overall mean (0.154 vs 0.134), and **expressed-gene magnitude (nonzero mean 1.96 vs 1.96)**.
Class-averaged distributional fidelity (lower is better):

| Method | MMD (ambient) | MMD (PCA) | W2 (ambient) | W2 (PCA) |
|--------|:-------------:|:---------:|:------------:|:--------:|
| **DDPM v1 (w=1)** | **0.008** | **0.010** | **0.160** | **5.62** |
| DDPM v2 (end-to-end) | 0.039 | 0.094 | 0.200 | 6.68 |
| cVAE | 0.130 | 0.101 | 0.238 | 5.69 |
| Gaussian-PCA | 0.090 | 0.015 | 0.206 | 5.78 |

**Biological validity:** for every canonical marker the two-sample KS test **fails to reject**
(p > 0.1) — generated and real marker expression are statistically indistinguishable (e.g.
LYZ in CD14+ monocytes 5.04 → 5.04; CD3D in T cells 2.24 → 2.08; MS4A1 in B cells 2.00 → 2.20;
PPBP in the rare Megakaryocytes 6.51 → 5.87). **Geometric structure:** generated cells
reproduce all 8 cell-type clusters with correct UMAP geometry.

**Zero-inflation ablation** (same backbone, vary only inference):

| Config | Zero% | NZ-mean | MMD-amb | MMD-PCA |
|--------|:-----:|:-------:|:-------:|:-------:|
| no gate | 16.6 | 0.45 | 0.391 | 0.522 |
| hurdle gate | 92.0 | 1.21 | 0.110 | 0.132 |
| learned gate + calibration | 92.2 | **1.96** | **0.008** | **0.010** |

> **Key empirical finding:** restoring sparsity is the single biggest gain (MMD 0.39 → 0.11).
> *Sparsity (dropout) modeling dominates distributional fidelity for scRNA diffusion*, more than
> any diffusion-backbone detail.

**v1 vs v2:** the end-to-end v2 recovers the expressed magnitude (1.975 vs real 1.961) **without
any calibration**, proving the v1 undershoot was a standardization artifact, not a fundamental
limit. v2 trails v1 on PCA-space distance (calibration matches marginals by construction) but is
the honest, fully-generative model.

### 5.2 Against real SOTA (scVI) and downstream utility

**Distributional fidelity** — scVI is strong; v1 ties it (partly by calibration), v2 loses:

| Method | MMD-amb | MMD-PCA |
|--------|:-------:|:-------:|
| scVI | 0.011 | 0.018 |
| DDPM v1 | 0.008 | 0.010 |
| DDPM v2 | 0.039 | 0.094 |

**Downstream augmentation** (few-shot cell-type classification, macro-F1 gain over real-only):

| K real/class | +scVI | +v1 | +v2 |
|:------------:|:-----:|:---:|:---:|
| 10 | **+0.155** | +0.127 | +0.129 |
| 25 | +0.098 | **+0.104** | +0.099 |
| 50 | +0.074 | +0.059 | +0.065 |

All three generators help substantially (largest in the scarcest regime; per-class F1 at K=10
shows the rarest class, Megakaryocytes, rescued from F1 = 0 → 1). **But scVI gives the best
macro-F1 gains** — the diffusion does not win the downstream task either.

### 5.3 Searching for a differentiated axis (controllable + perturbation)

- **Controllable generation (§10):** CFG-strength control is *inert* on PBMC-3k (cell types are
  too separable; purity already 96% at w=0). Per-cell SDEdit counterfactual editing *does* exist
  in the raw diffusion (it transfers identity while preserving source structure above a random
  floor, 0.44 vs 0.33), but the **fidelity machinery (gate + calibration) erases that
  editability** — a genuine *fidelity ↔ editability tension*, but not a SOTA-beating result.
- **Perturbation prediction (§11, Kang IFN-β vs scGen):** the model correctly predicts the
  interferon response (ISG15/IFIT3/CXCL10 up) in held-out cell types it never saw stimulated —
  qualitatively a success. But across 3 held-out types, on the headline mean-change metric
  **R²-ΔDEG, scGen wins 2 of 3** (diffusion wins only the largest type, CD4, where holding out
  its stim cells handicaps scGen's latent delta):

  | Held-out | scGen R²-ΔDEG | diffusion R²-ΔDEG |
  |----------|:-------------:|:-----------------:|
  | CD4 T (5560) | 0.11 | **0.70** |
  | B cells (1316) | **0.88** | 0.39 |
  | NK cells (855) | **0.90** | 0.10 |

  A distribution metric *designed to favor* the diffusion (per-DEG 1-Wasserstein + variance
  match) **still favors scGen** (scGen wins variance-match 3/3 and W1 2/3; the diffusion is
  sometimes worse than the no-op baseline). The hypothesis "diffusion captures response
  heterogeneity better" is **refuted** on this benchmark.

### 5.4 Discussion, limitations, and honest verdict

**What the project genuinely contributes.**
1. A **validated, reproducible transfer** of vision diffusion (DDPM + CFG + DDIM) to scRNA-seq,
   with a clear diagnosis of *why* it breaks and the minimal fixes that repair it.
2. A reusable empirical finding: **sparsity/dropout modeling dominates distributional fidelity**
   for scRNA diffusion (none → hurdle is the biggest single gain).
3. A clean **v1-vs-v2** distinction between matching marginals *by injection* (calibration) vs
   *by generation* (expressed-only scaling), and an explicit **fidelity ↔ editability tension**.
4. **Rigorously-established negative results:** across five axes — distributional fidelity,
   augmentation utility, controllable generation, perturbation mean-prediction, and perturbation
   distribution-prediction — the diffusion is *competitive* but shows **no demonstrable advantage
   over SOTA (scVI / scGen)**, even on a metric purpose-built to favor it.

**Honest assessment of novelty.** The components (DDPM/CFG/DDIM, z-scoring, x₀-prediction/
clipping, a ZINB-style dropout head, quantile normalization) are established techniques; the
contribution is their *integration and diagnosis* for this modality plus the negative findings —
not a new method or a SOTA win. Notably, the core "separate dropout from magnitude" principle is
already embodied in scVI's ZINB likelihood, so confirming it in a diffusion framework is
incremental.

**Limitations.**
- **Single small datasets** (PBMC-3k; Kang held-out types) and **single seed** — no multi-seed
  confidence intervals.
- **Calibration injects marginals**, so v1's near-zero ambient metrics are partly by
  construction; the defensible evidence is the PCA-space metrics, UMAP geometry, and marker
  co-expression (which the diffusion genuinely supplies).
- The perturbation advantage is **population-size dependent** and the per-cell edit is **noisier
  on small held-out populations** — and degrades the distribution rather than capturing
  heterogeneity as hypothesized.

**Future improvements.**
- An **end-to-end zero-inflated likelihood** (e.g. a diffusion with a ZINB/hurdle decoding head)
  that learns dropout *and* magnitude jointly, removing the post-hoc calibration entirely.
- **Multi-dataset** evaluation (sci-Plex, Norman/Perturb-seq) against **modern** perturbation
  models (CellOT, CPA, biolord, diffusion/flow-based methods), not just scGen, using standard
  distribution metrics (energy distance / E-distance from the scPerturb benchmark).
- Tasks where **per-cell heterogeneity is biologically load-bearing** (bimodal responses,
  differentiation trajectories), where a sampling-based generator should structurally beat a
  mean-shift VAE — the one regime not yet tested.

---

## 6. References

**Methods / models**
1. J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
2. J. Ho, T. Salimans. *Classifier-Free Diffusion Guidance.* arXiv:2207.12598, 2022.
3. J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models (DDIM).* ICLR 2021.
4. C. Meng et al. *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential
   Equations.* ICLR 2022. (basis of the counterfactual-editing experiment)
5. R. Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.

**Single-cell generative models (baselines / context)**
6. R. Lopez, J. Regier, M. Cole, M. Jordan, N. Yosef. *Deep generative modeling for single-cell
   transcriptomics (scVI).* Nature Methods, 15(12):1053–1058, 2018.
7. M. Lotfollahi, F. A. Wolf, F. J. Theis. *scGen predicts single-cell perturbation responses.*
   Nature Methods, 16:715–721, 2019.
8. C. H. Grønbech et al. *scVAE: variational auto-encoders for single-cell gene expression data.*
   Bioinformatics, 36(16):4415–4422, 2020.
9. E. Luo, M. Hao, L. Wei, X. Zhang. *scDiffusion: conditional generation of high-quality
   single-cell data using diffusion model.* Bioinformatics, 40(9):btae518, 2024.
10. L. Bini, S. Marchand-Maillet. *LapDDPM: A Conditional Graph Diffusion Model for scRNA-seq
    Generation.* arXiv:2506.13344, 2025.

**Datasets**
11. G. X. Y. Zheng et al. *Massively parallel digital transcriptional profiling of single cells
    (PBMC-3k).* Nature Communications, 8:14049, 2017.
12. H. M. Kang et al. *Multiplexed droplet single-cell RNA-sequencing using natural genetic
    variation (IFN-β stimulation).* Nature Biotechnology, 36:89–94, 2018.
13. 10x Genomics. *3k PBMCs from a Healthy Donor.* https://www.10xgenomics.com/datasets

**Software / codebases**
14. F. A. Wolf, P. Angerer, F. J. Theis. *SCANPY: large-scale single-cell gene expression data
    analysis.* Genome Biology, 19:15, 2018. — preprocessing, datasets, UMAP.
15. scvi-tools (https://scvi-tools.org) — scVI baseline implementation.
16. pertpy (https://pertpy.readthedocs.io) — Kang 2018 dataset loader / source URL.
17. PyTorch; scikit-learn; SciPy; anndata; umap-learn — core numerical/ML libraries.

**This project's code:** all methods, experiments, and figures are implemented in the submitted
repository (`main.py`, `src/`, `experiments/`, `configs/`); see `README.md` (§1–13) for the full
write-up and `DATA.md` for the data dictionary.
