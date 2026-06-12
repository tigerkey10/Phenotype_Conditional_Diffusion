# Data dictionary

---

# Understanding the data

## 0.1 What is this data? (explained for a vision generative-modeling researcher)

This project ports **image-style diffusion (DDPM + classifier-free guidance)** to
**single-cell RNA-seq (scRNA-seq)**. The fastest way to understand the data is by analogy to
image generation — and, just as importantly, by where the analogy *breaks*.

**The 1:1 mapping (what stays the same):**

| Image generation | scRNA-seq here |
|------------------|----------------|
| an image (a sample) | a **cell** (a sample) |
| pixel vector `x ∈ ℝ^{H·W·C}` | **gene-expression vector** `x ∈ ℝ^G` (G = 1,000–2,000 genes) |
| one pixel intensity | one gene's expression level |
| class label (e.g. ImageNet category) | **cell-type label** (T cell, B cell, monocyte, …) |
| class-conditional generation + CFG | identical: condition the denoiser on cell type, guide with w |
| FID / Inception distance | **MMD / 2-Wasserstein** (we literally use the Fréchet/Gaussian-W2 = FID formula in PCA space) |
| t-SNE / qualitative mode-coverage check | **UMAP** of real vs generated |
| image editing / I2I translation / SDEdit | **perturbation editing** (control → stimulated cell, §11) |

So conditioning, guidance, DDIM sampling, and FID-style evaluation transfer verbatim.

**Where the analogy breaks (and why this is the whole research problem):**

1. **No spatial grid.** A gene-expression vector is a *tabular / set* feature vector, not a 2D
   image. Genes have no locality or translation structure — column order is arbitrary. So the
   denoiser is an **MLP, not a U-Net/CNN**; there is no convolution, no patch structure.
2. **The marginal is a spike at zero, not a natural-image histogram.** Natural images densely
   fill their value range and are locally smooth. Here **~93% of every vector is exactly 0**
   (genes are "off" or undetected — *dropout*), with a thin non-negative positive tail (values
   ≈ [0, 7] after normalization). Think of generating *extremely sparse, non-negative images
   where almost every pixel is hard zero* — a delta-spike-plus-slab marginal per dimension.
3. **Per-dimension variance is tiny and the data is not unit-scaled.** Standard DDPM implicitly
   assumes data ≈ unit variance (images are scaled to [−1, 1]). Log-norm expression has small,
   gene-specific variance dwarfed by the spike at 0. Feeding it raw, the unit-variance forward
   noise **drowns the signal** → a literal DDPM produces garbage (generated values ~200× the
   real scale). This is the first thing that must be fixed (`README.md` §5).
4. **Exact zeros are unreachable by continuous diffusion.** A Gaussian reverse process emits
   continuous values; it cannot place a point mass at exactly 0. Reproducing the 93% sparsity
   needs an explicit **on/off (Bernoulli) gate** on top of the continuous magnitude — a
   *spike-and-slab* / hurdle structure with no real analogue in standard image diffusion.

**One-line intuition:** treat each cell as a **1,000-dimensional, permutation-arbitrary,
~93%-zero, non-negative feature vector** with a class label — *not* a picture. Class-conditional
diffusion machinery carries over; the image-specific assumptions (spatial locality, dense
roughly-Gaussian-after-scaling pixels) do not — and bridging that gap is exactly what
`README.md` §5–§11 is about.

The underlying object is still a counts matrix (rows = cells, columns = genes, entries =
transcript counts), with a clustering-derived cell-type label per cell:

```
                gene_1  gene_2  gene_3   ...   gene_G
   cell_1   [     0       5       0     ...      2   ]      label: CD4 T cell
   cell_2   [     3       0       0     ...      0   ]      label: B cell
     ...
```

## 0.2 The two datasets

**PBMC-3k** — *the cell-type generation dataset* (project core, `README.md` §1–10).
~2,700 **P**eripheral **B**lood **M**ononuclear **C**ells from one healthy donor (10x
Genomics; the standard scanpy tutorial dataset). After filtering, **2,638 cells** annotated
into **8 immune cell types**. There is **no perturbation** — the task is purely *conditional
generation*: "generate a synthetic cell of type X". The 8 types and their canonical **marker
genes** (genes switched on in that type, used to validate generated cells):

| Cell type | Role (brief) | Marker gene(s) | Rarity in data |
|-----------|--------------|----------------|----------------|
| CD4 T cells | helper T lymphocytes | CD3D | most common |
| CD14+ Monocytes | classical monocytes (innate) | LYZ | common |
| B cells | antibody-producing lymphocytes | MS4A1 | medium |
| CD8 T cells | cytotoxic T lymphocytes | CD3D, CD8A | medium |
| NK cells | natural killer (innate cytotoxic) | NKG7, GNLY | medium |
| FCGR3A+ Monocytes | non-classical monocytes | FCGR3A, LYZ | uncommon |
| Dendritic cells | antigen-presenting | — | rare (~37) |
| Megakaryocytes | platelet precursors | PPBP | rarest (~15) |

Exact PBMC counts (2,638 cells total): CD4 T 1,144 · CD14+ Mono 480 · B 342 · CD8 T 316 ·
NK 154 · FCGR3A+ Mono 150 · Dendritic 37 · Megakaryocytes 15. The strong class imbalance
(CD4 T ≈ 76× Megakaryocytes) is deliberate signal: it lets us test **rare-cell-type**
generation and augmentation.

**Kang et al. 2018** — *the perturbation dataset* (`README.md` §11).
**24,673** PBMCs, the same 8 cell types, but now each cell is in one of **two conditions**:

- `ctrl` — unstimulated control, and
- `stim` — stimulated with **IFN-β** (interferon-beta), a cytokine that triggers a strong,
  well-characterized antiviral immune response.

IFN-β stimulation switches on **interferon-stimulated genes (ISGs)** — the perturbation
"signature" the model must learn to predict. The top up-regulated genes (mean log-expression
increase ctrl→stim) are:

```
ISG15 (+4.0)  ISG20 (+2.9)  IFIT3 (+2.8)  IFIT2 (+2.0)
CXCL10 (+2.0)  RSAD2 (+1.7)  IFITM3 (+1.5)
```

This dataset supports a **counterfactual / perturbation-response** task — the scRNA-seq
analogue of *attribute editing / image-to-image translation* (think "add glasses to this
face"): hold out one cell type's `stim` cells, then predict "what would this control cell look
like if stimulated?", having learned the ctrl→stim shift only from the *other* cell types. We
implement it with **SDEdit** (noise the control cell partway, denoise toward "stim"); the
baseline (scGen) is latent vector arithmetic, the scRNA-seq cousin of latent-space attribute
vectors in GAN/VAE image editing.

## 0.3 How raw data becomes model input (preprocessing)

Both datasets go through the same standard pipeline (`src/data.py`), turning raw counts into
the `X` matrix the model trains on:

| Step | What | Why |
|------|------|-----|
| filter | drop cells <200 genes, genes in <3 cells | remove empty droplets / never-seen genes |
| `normalize_total(1e4)` | rescale each cell to 10,000 total counts | remove sequencing-depth differences (CP10K) |
| `log1p` | `x → log(1+x)` | tame the heavy right tail; stabilize variance |
| HVG selection | keep top **G** highly variable genes (1,000 PBMC / 2,000 Kang) | focus on informative, cell-type-distinguishing genes |

This is the rough equivalent of mapping pixels to `[−1, 1]` before training an image diffusion
model — **except it does not make the data Gaussian-like.** The result `X` ∈ ℝ^{cells × G} of
**log-normalized** expression is still ~93% zeros, values roughly in [0, 7], most genes near 0
with a thin positive tail. *That sparse, zero-inflated, low-magnitude shape — so unlike the
dense, scaled pixel arrays diffusion was built for — is exactly why a vanilla vision-style
diffusion model breaks on it*, and what `README.md` §5 fixes.

---

## Raw downloads — `data/`

Regenerated automatically on first run; not committed (see `.gitignore`).

| File | Source | Contents |
|------|--------|----------|
| `data/pbmc3k_raw.h5ad` | `scanpy.datasets.pbmc3k()` | 2,700 × 32,738 **raw counts**, no labels |
| `data/pbmc3k_processed.h5ad` | `scanpy.datasets.pbmc3k_processed()` | annotated version; `obs['louvain']` = the 8 cell-type labels |
| `data/kang_2018.h5ad` | figshare `ndownloader.figshare.com/files/34464122` | Kang 2018 PBMC IFN-β: 24,673 × 15,706 **raw counts**; `obs['cell_type']` (8 types), `obs['label']` ∈ {ctrl, stim} |

---

