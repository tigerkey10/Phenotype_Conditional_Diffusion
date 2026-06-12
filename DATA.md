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

