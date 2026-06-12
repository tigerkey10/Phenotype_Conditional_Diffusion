# Data dictionary

Schema and provenance of every data artifact in this repo. Binary files (`.h5ad`,
`.npz`, `.pt`) can't carry inline comments, so this is their reference. See `README.md`
for the science; this file is purely "what is in each file".

---

## 1. Raw downloads — `data/`

Regenerated automatically on first run; not committed (see `.gitignore`).

| File | Source | Contents |
|------|--------|----------|
| `data/pbmc3k_raw.h5ad` | `scanpy.datasets.pbmc3k()` | 2,700 × 32,738 **raw counts**, no labels |
| `data/pbmc3k_processed.h5ad` | `scanpy.datasets.pbmc3k_processed()` | annotated version; `obs['louvain']` = the 8 cell-type labels |
| `data/kang_2018.h5ad` | figshare `ndownloader.figshare.com/files/34464122` | Kang 2018 PBMC IFN-β: 24,673 × 15,706 **raw counts**; `obs['cell_type']` (8 types), `obs['label']` ∈ {ctrl, stim} |

---

## 2. Preprocessed caches — `*/…_hvg.h5ad`

Produced by `src/data.py:preprocess` (PBMC) or `experiments/pert_build_train.py:build_cache`
(Kang). Same AnnData schema for both:

| Field | Type / shape | Meaning |
|-------|--------------|---------|
| `X` | float32, (n_cells, G) | **log-normalized** expression over the G highly-variable genes (CP10K → log1p → HVG). `G=1000` for PBMC, `G=2000` for Kang. This is the model's input/eval space. |
| `var_names` | (G,) | the selected HVG gene symbols (marker genes force-included for PBMC) |
| `obs['cell_type']` | categorical | human-readable cell type |
| `obs['label']` | int | **integer condition/class label** — PBMC: 0–7 = cell type (index into `uns['classes']`); Kang: 0 = ctrl, 1 = stim |
| `uns['classes']` | (n_classes,) | string names; `label` indexes this. PBMC order: `['B cells','CD14+ Monocytes','CD4 T cells','CD8 T cells','Dendritic cells','FCGR3A+ Monocytes','Megakaryocytes','NK cells']` |
| `raw` | AnnData | (PBMC only) all-gene log-norm, so marker genes survive HVG selection |

Files: `runs/outputs/pbmc3k_hvg.h5ad`, `runs/outputs_kang/kang_hvg.h5ad`,
`runs/outputs_kang_{B_cells,NK_cells}/kang_hvg.h5ad` (identical data, one per held-out experiment).

---

## 3. Model checkpoints — `*/checkpoints/ddpm_best.pt`

`torch.load` returns a dict. The non-`model` entries are everything the sampler needs to
map the network's standardized output back to real expression and to restore sparsity —
so a checkpoint is fully self-contained.

| Key | Type / shape | Meaning |
|-----|--------------|---------|
| `model` | state_dict | EMA weights of `ResidualMLPDenoiser` (incl. `gate_proj.*` if a gate head was trained) |
| `n_genes`, `n_classes` | int | G and number of label classes |
| `model_cfg` | dict | `hidden, n_blocks, time_dim, class_dim, dropout, gate_head` — used to rebuild the architecture |
| `scaler_mean`, `scaler_std` | (G,) | per-gene standardization stats (all-cell or expressed-only); inverted at sampling |
| `x0_lo`, `x0_hi` | (G,) | per-gene clip bounds for x0 during DDIM (stability) |
| `zero_rate` | (n_classes, G) | per-class/gene dropout (zero) rate — the hurdle gate's MLE Bernoulli |
| `mag_quant` | (n_classes, G, 64) | per-class/gene quantiles of real nonzero expression — for magnitude calibration |
| `val_mmd`, `epoch` | float, int | best validation MMD and the epoch it was saved at |

Which design each checkpoint encodes: `runs/outputs/` & `runs/outputs_v1_calibrated/` = v1 (all-cell
scaling + calibration); `runs/outputs_end2end/` = v2 (expressed-only scaling, no calibration);
`runs/outputs_kang*/` = perturbation models (condition-conditioned, one per held-out cell type).

---

## 4. Generated samples — `*/samples/*.npz`

`np.load` gives a dict with keys `class_0 … class_{n_classes-1}`; each value is a
float32 array `(n_per_class, G)` of generated **log-norm** cells for that class, already
de-standardized + gated (+ calibrated for v1). File name encodes the guidance scale,
e.g. `ddpm_w1.0.npz` (w=1). `runs/outputs_scvi/samples/scvi.npz` has the same layout.

---

## 5. Metrics & figures

| File | Contents |
|------|----------|
| `*/metrics/metrics.json` | `{classes, methods:{<method>:{per_class:{<type>:{mmd_ambient,mmd_pca,w2_ambient,w2_pca_gauss,w2_pca_sliced}}, aggregate:{…}}}, markers:{guidance_scale, results:{<gene>:{expected_high_in, in_hvg, per_class:{<type>:{ks_stat,ks_p,real_mean,gen_mean}}}}}}` |
| `*/metrics/summary.csv` | one row per method, class-averaged MMD/W2 columns |
| `*/figures/umap_real_vs_ddpm.png` | joint UMAP, real vs generated |
| `figures/umap_v1_vs_v2.png` | v1-vs-v2 shared-embedding UMAP (§8) |

---

## 6. Run logs

`runs/outputs_*_run.log`, `runs/outputs_kang_train.log`, `runs/outputs_kang_more.log` — stdout of the
corresponding training/eval runs (loss curves, val MMD, final metrics). Regenerable.
