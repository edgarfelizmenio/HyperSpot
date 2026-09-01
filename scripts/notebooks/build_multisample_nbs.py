import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

DEMO = "/workspace/HyperSpot/demo"

# ---- shared config block: BOTH sections, one flat work list ----
SAMPLES_BLOCK = '''# Both sections are processed in one run. One representative (largest) section per
# donor, priority-ordered by spot count.
# SKIPPED donors:
#   PD47171 — already processed as the existing "Human Kidney Cancer Interface"/"Core" projects.
#   PD47465 — scRNA reference too sparse (only 11,565 cells; Epi_non-PT=1, EC=8, Epi_PT/Fibro=7,
#             Mast=6), which makes Spotiphy marker selection divide by zero. Dropping it keeps every
#             remaining donor on a pure per-patient reference (interface: 7 -> 6 donors; core unaffected).
SKIP = {"PD47171", "PD47465"}
SECTIONS = {
    # section: (merged source file, [(patient, sample, spots), ...])
    "Interface": ("visium_merge_inter_upload.h5ad", [
        ("PD43824", "6800STDY12499406", 3352),
        ("PD47465", "6800STDY12499508", 2831),   # SKIP — scRNA too sparse (11,565 cells; Epi_non-PT=1)
        ("PD45814", "6800STDY12499502", 2700),
        ("PD45815", "6800STDY12499504", 2670),
        ("PD45816", "6800STDY12499507", 2145),
        ("PD47512", "6800STDY12499409", 2085),
        ("PD47171", "6800STDY12499411", 2048),   # SKIP (existing project)
    ]),
    "Core": ("visium_merge_core_upload.h5ad", [
        ("PD47512", "6800STDY12499408", 3935),
        ("PD47171", "6800STDY12499410", 3008),    # SKIP (existing project)
        ("PD45816", "6800STDY12499413", 2102),
        ("PD45814", "6800STDY12499412", 1047),
        ("PD43948", "6800STDY12499407",  832),
    ]),
}
def project_name(section, patient, sample):
    return f"Kidney Cancer {section} {patient} {sample}"
# flat work list across BOTH sections (skip PD47171)
TODO = [(sec, p, s, n) for sec, (src, man) in SECTIONS.items()
        for (p, s, n) in man if p not in SKIP]
print(f"{len(TODO)} samples across both sections:")
for sec, p, s, n in TODO:
    print(f"  {project_name(sec, p, s)}  ({n} spots)")'''


# Python-side copy of the manifest, used only to generate the static eval-command markdown in nb2.
SKIP_PY = {"PD47171", "PD47465"}
SECTIONS_PY = {
    "Interface": ("visium_merge_inter_upload.h5ad", [
        ("PD43824", "6800STDY12499406", 3352), ("PD47465", "6800STDY12499508", 2831),
        ("PD45814", "6800STDY12499502", 2700), ("PD45815", "6800STDY12499504", 2670),
        ("PD45816", "6800STDY12499507", 2145), ("PD47512", "6800STDY12499409", 2085),
        ("PD47171", "6800STDY12499411", 2048)]),
    "Core": ("visium_merge_core_upload.h5ad", [
        ("PD47512", "6800STDY12499408", 3935), ("PD47171", "6800STDY12499410", 3008),
        ("PD45816", "6800STDY12499413", 2102), ("PD45814", "6800STDY12499412", 1047),
        ("PD43948", "6800STDY12499407", 832)]),
}


def kernel(nb):
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


# =====================================================================================
# Notebook 0 — Data Preparation (HyperSpot env), both sections
# =====================================================================================
c0 = []
c0.append(md(r"""# Human Kidney Cancer — Multi-Sample 0: Data Preparation

**Environment: `HyperSpot`.**  Processes **both the Interface (7 donors) and Core (5 donors)**
sections in a single run.

Generalizes the single-donor data-prep to **all donors** for the cross-donor conservation analysis.
From each section's merged source it writes, **per sample**, a paired scRNA reference + ST AnnData
(+ H&E image) into a Spotiphy project folder.

- One representative (largest) section per donor; **PD47171 is skipped** in both sections (it is the
  existing single-donor project). Sections share the Visium grid, so each sample is its own project.
- scRNA is subset to each sample's patient; gene-ID harmonization uses the same SYMBOL-collision
  resolution as the single-donor pipeline (verified identical across samples).
- Plotting is skipped (batch prep)."""))

c0.append(md("Imports — scanpy plus the HyperSpot preprocessing helpers used by the single-donor prep."))
c0.append(code(r"""import scanpy as sc
from pathlib import Path
from hyperspot.preprocessing import common, image, kidneycancer"""))

c0.append(md("Sample manifest — both sections, one flat work list (shared across all three notebooks)."+' **PD47465 is dropped** — its scRNA reference has too few cells (11,565 total; Epi_non-PT=1, EC=8, Epi_PT/Fibro=7, Mast=6), which breaks Spotiphy marker selection; every remaining donor keeps a pure per-patient reference.'))
c0.append(code(SAMPLES_BLOCK))

c0.append(md("""scRNA source (shared by both sections) and the Ensembl gene annotation (cached).

**Why drop / rename:** a handful of Ensembl IDs collide on a **SYMBOL that another gene already
carries**, and `use_ensembl_gene_ids` requires unique symbols. Each colliding pair is resolved by:
- **drop** the redundant member (GGT1/MATR3/TBCE: one ID is unnamed or a non-canonical duplicate), or
- **rename** the member whose *true* gene identity is distinct so we keep both genes
  (`ENSG00000284024` shares SYMBOL HSPA14 but is really **MSANTD7**; `ENSG00000269226` shares
  SYMBOL TMSB15B but is really **TMSB15C**).

The ST panel (36,601 genes) and scRNA reference are shared across all samples, so these collisions —
and this resolution — are identical for every sample."""))
c0.append(code(r"""data_dir = Path("data_repo/Human Kidney Cancer")
adata_sc_all = sc.read_h5ad(data_dir / "RCC_upload_final_raw_counts.h5ad")
geneset = common.get_annotation(host="dec2021.archive.ensembl.org",
                                cache_filename="GRCh38-2020-A_annotations.json")

# each ID shares its ST SYMBOL with another gene; resolve to keep symbols unique
drop_cols = [
    "ENSG00000286070",  # shares SYMBOL GGT1  (unnamed in Ensembl)  -> keep ENSG00000100031
    "ENSG00000280987",  # shares SYMBOL MATR3 (non-canonical dup)   -> keep ENSG00000015479
    "ENSG00000285053",  # shares SYMBOL TBCE  (non-canonical dup)   -> keep ENSG00000284770
]
rename_cols = {
    "ENSG00000284024": "MSANTD7",   # shares SYMBOL HSPA14  -> rename to its true symbol (keep gene)
    "ENSG00000269226": "TMSB15C",   # shares SYMBOL TMSB15B -> rename to its true symbol (keep gene)
}
print("scRNA:", adata_sc_all.shape)"""))

c0.append(md("""Per-sample extraction helper. `extract_st_patient_data` pulls a donor (all its sections)
from the given merged ST; we subset to the one chosen `sample` and prune `uns['spatial']` to that
sample's H&E library so coordinates/image are unambiguous."""))
c0.append(code(r"""def extract_sample(adata_st_all, patient, sample):
    a_sc = kidneycancer.extract_sc_patient_data(adata_sc_all, patient)
    a_st = kidneycancer.extract_st_patient_data(adata_st_all, patient)
    a_st = a_st[a_st.obs["sample"] == sample].copy()
    for lib in list(a_st.uns["spatial"].keys()):      # keep only this sample's H&E library
        if sample not in lib:
            a_st.uns["spatial"].pop(lib, None)
    return a_sc, a_st"""))

c0.append(md("""Process **both sections**: load each section's merged ST once, then per sample verify
the SYMBOL-collision conflicts are covered by the shared drop/rename set, harmonize gene IDs, and write
the scRNA reference + ST + H&E image into `spotiphy_project_files/<project name>/`."""))
c0.append(code(r"""resolved = set(drop_cols) | set(rename_cols)
for section, (source_st, manifest) in SECTIONS.items():
    print(f"\\n########## {section} ##########")
    adata_st_all = sc.read_h5ad(data_dir / source_st)
    for patient, sample, n in manifest:
        if patient in SKIP:
            continue
        proj = project_name(section, patient, sample)
        out = Path("spotiphy_project_files") / proj
        out.mkdir(parents=True, exist_ok=True)

        a_sc, a_st = extract_sample(adata_st_all, patient, sample)

        # consistency: every duplicate-SYMBOL pair must be covered by the shared resolution
        conf = kidneycancer.find_conflicting_genes(a_sc, a_st, geneset)
        if len(conf):
            bad = [sym for sym, grp in conf[conf["in_sc"]].groupby("SYMBOL", observed=True)
                   if not (set(grp.index) & resolved)]
            assert not bad, f"{proj}: new SYMBOL collision {bad} — extend drop_cols/rename_cols."

        a_sc, a_st = kidneycancer.use_ensembl_gene_ids(a_sc, a_st, geneset,
                                                       drop_cols=drop_cols, rename_cols=rename_cols)
        img = image.get_image(a_st, scale=True)
        a_sc.write(out / "Human Kidney Cancer scRNA Ref.h5ad")
        a_st.write(out / f"{proj}.h5ad")
        img.save(out / f"{proj}.tif")
        print(f"[{proj}] scRNA {a_sc.shape} + ST {a_st.shape}")
print(f"\\nData preparation done for {len(TODO)} samples (both sections).")"""))

nb0 = kernel(new_notebook(cells=c0))
nbf.write(nb0, f"{DEMO}/Human Kidney Cancer Multi-Sample 0 - Data Preparation.ipynb")


# =====================================================================================
# Notebook 1 — Spotiphy Deconvolution + Decomposition (Spotiphy env), both sections
# =====================================================================================
c1 = []
c1.append(md(r"""# Human Kidney Cancer — Multi-Sample 1: Spotiphy Deconvolution

**Environment: `Spotiphy`** (this notebook `import spotiphy`, which exists only in the Spotiphy conda
env — run it with that kernel; notebooks 0 and 2 use the HyperSpot env).

Runs Spotiphy on **every sample of both sections** through **deconvolution + decomposition only**.
All plotting is skipped **except the cell-type proportion heatmaps**; nucleus segmentation is still
*computed* (its per-spot cell counts feed decomposition, matching the single-donor pipeline) but not
plotted; the pseudo-image / imputation section is omitted."""))

c1.append(md("Imports and single-GPU / TensorFlow setup (same as the single-donor Spotiphy notebook)."))
c1.append(code(r"""import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import spotiphy, pickle
import numpy as np, pandas as pd, scanpy as sc, torch, cv2
import matplotlib as mpl
from pathlib import Path
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))"""))

c1.append(md("Sample manifest — both sections (identical to Multi-Sample 0). Edit `TODO` to run a subset."+' **PD47465 is dropped** — its scRNA reference has too few cells (11,565 total; Epi_non-PT=1, EC=8, Epi_PT/Fibro=7, Mast=6), which breaks Spotiphy marker selection; every remaining donor keeps a pure per-patient reference.'))
c1.append(code(SAMPLES_BLOCK))

c1.append(md("""Deconvolution + decomposition, per sample across both sections. `plot=False` on
`estimation_proportion` suppresses the training-loss plot; **the cell-type proportion spatial heatmaps
are drawn and saved** (`Spotiphy_deconvolution.jpg`). Segmentation is computed for per-spot cell counts
but not plotted."""))
c1.append(code(r"""key_type = "broad_type"
device = "cuda" if torch.cuda.is_available() else "cpu"

for section, patient, sample, n in TODO:
    proj = project_name(section, patient, sample)
    folder = Path("spotiphy_project_files") / proj
    print(f"\\n===== {proj} =====")

    # --- load prepared inputs ---
    adata_sc = sc.read_h5ad(folder / "Human Kidney Cancer scRNA Ref.h5ad")
    adata_st = sc.read_h5ad(folder / f"{proj}.h5ad")
    img = cv2.imread(str(folder / f"{proj}.tif"))
    adata_st.var_names_make_unique()
    adata_st.obsm["spatial"] = adata_st.obsm["spatial"].astype(np.int32)
    type_list = sorted(list(adata_sc.obs[key_type].unique().astype(str)))

    # --- marker selection + single-cell reference (heatmap plot skipped) ---
    adata_sc, adata_st = spotiphy.initialization(adata_sc, adata_st, verbose=1)
    marker_gene_dict = spotiphy.sc_reference.marker_selection(
        adata_sc, key_type=key_type, return_dict=True,
        n_select=50, threshold_p=0.1, threshold_fold=1.5, q=0.15)
    marker_gene = [g for t in type_list for g in marker_gene_dict[t]]
    pd.DataFrame({"gene": marker_gene,
                  "label": [t for t in type_list for _ in marker_gene_dict[t]]}
                 ).to_csv(folder / "marker_gene.csv")
    adata_sc_marker = adata_sc[:, marker_gene]
    adata_st_marker = adata_st[:, marker_gene]
    sc_ref = spotiphy.construct_sc_ref(adata_sc_marker, key_type=key_type)

    # --- cellular proportion estimation (loss plot skipped) ---
    X = np.array(adata_st_marker.X)
    cell_proportion = spotiphy.deconvolution.estimation_proportion(
        X, adata_sc_marker, sc_ref, type_list, key_type,
        n_epoch=8000, plot=False, batch_prior=1, device=device)
    adata_st.obs[type_list] = cell_proportion
    np.save(folder / "proportion.npy", cell_proportion)
    np.savetxt(folder / "proportion.csv", adata_st.obs[type_list], delimiter=",")

    # --- KEEP: cell-type proportion heatmaps ---
    vmax = np.quantile(adata_st.obs[type_list].values, 0.98, axis=0); vmax[vmax < 0.05] = 0.05
    with mpl.rc_context({"figure.figsize": [3, 5], "figure.dpi": 200, "xtick.labelsize": 0}):
        ax = sc.pl.spatial(adata_st, cmap="viridis", color=type_list, img_key="hires",
                           vmin=0, vmax=list(vmax), size=1.3, show=False, ncols=5, alpha_img=0.4)
        ax[0].get_figure().suptitle(proj, y=1.02)
        ax[0].get_figure().savefig(folder / "Spotiphy_deconvolution.jpg", bbox_inches="tight")
    mpl.pyplot.show()

    # --- segmentation: compute per-spot cell counts (plot skipped) ---
    seg_dir = folder / "segmentation"; seg_dir.mkdir(exist_ok=True)
    segmentation = spotiphy.segmentation.Segmentation(
        img[:, :, [2, 1, 0]], adata_st.obsm["spatial"],
        n_tiles=(32, 32, 1), out_dir=str(seg_dir), enhancement=True)
    segmentation.segment_nucleus(save=True)
    n_cell_df = segmentation.n_cell_df
    pd.DataFrame(n_cell_df).to_csv(seg_dir / "n_cell_per_spot.csv", index=False)

    # --- decomposition to single-cell resolution (segmentation-based n_cell) ---
    adata_sc_orig = sc.read_h5ad(folder / "Human Kidney Cancer scRNA Ref.h5ad")
    adata_st_orig = sc.read_h5ad(folder / f"{proj}.h5ad")
    adata_st_orig.var_names_make_unique()
    adata_decomp = spotiphy.deconvolution.decomposition(
        adata_st_orig, adata_sc_orig, key_type, cell_proportion,
        save=True, out_dir=str(folder) + "/", verbose=1,
        spot_location=adata_st_orig.obsm["spatial"], filtering_gene=True,
        n_cell=n_cell_df["cell_count"].values, filename="ST_decomposition.h5ad")

    # attach gene symbols and re-save (as in the single-donor notebook)
    adata_decomp = sc.read_h5ad(folder / "ST_decomposition.h5ad")
    adata_decomp.var["SYMBOL"] = adata_sc.var["name"]
    adata_decomp.write(folder / "ST_decomposition.h5ad")
    print(f"[{proj}] decomposition saved -> {folder / 'ST_decomposition.h5ad'}")
print(f"\\nAll {len(TODO)} samples (both sections) deconvolved + decomposed.")"""))

nb1 = kernel(new_notebook(cells=c1))
nbf.write(nb1, f"{DEMO}/Human Kidney Cancer Multi-Sample 1 - Spotiphy Deconvolution.ipynb")


# =====================================================================================
# Notebook 2 — HyperSpot Pipeline up to hypergraph (HyperSpot env) + eval commands
# =====================================================================================
c2 = []
c2.append(md(r"""# Human Kidney Cancer — Multi-Sample 2: HyperSpot Pipeline

**Environment: `HyperSpot`.**  Runs **every sample of both sections** up to hypergraph construction
and saves it. Model training + 10-seed evaluation is **not** run here — the last cell prints one
`perform-model-evaluation` command per sample (priority-ordered) to launch individually, so the
shared GPU isn't tied up by one long batch job. Plotting is skipped."""))

c2.append(md("Imports (HyperSpot pipeline helpers)."))
c2.append(code(r"""import scanpy as sc
from pathlib import Path
from hyperspot.metadata import ProjectMetadata
from hyperspot.preprocessing import common, singlecell, visium
from hyperspot.utils import save_orig_st_data, save_decomposed_st_data"""))

c2.append(md("Sample manifest — both sections (identical to the earlier notebooks)."+' **PD47465 is dropped** — its scRNA reference has too few cells (11,565 total; Epi_non-PT=1, EC=8, Epi_PT/Fibro=7, Mast=6), which breaks Spotiphy marker selection; every remaining donor keeps a pure per-patient reference.'))
c2.append(code(SAMPLES_BLOCK))

c2.append(md("""Pipeline per sample across both sections, up to and including hypergraph creation.
`ProjectMetadata` is a process-wide singleton, so we clear it before each sample and `create` a fresh
project — otherwise a later sample would silently reuse the first project's metadata. Config matches
the single-donor pipeline (HVG/`n_features`=3000, `num_neighbor_spots`=7, `n_hyperedge_feats`=3000)."""))
c2.append(code(r"""N_TOP_GENES = 3000
N_FEATURES  = 3000

for section, patient, sample, n in TODO:
    proj = project_name(section, patient, sample)
    print(f"\\n===== {proj} =====")
    spotiphy_dir = Path("spotiphy_project_files") / proj
    project_dir  = Path("hyperspot_project_files") / proj
    project_dir.mkdir(parents=True, exist_ok=True)

    # load Spotiphy outputs
    adata_decomp = sc.read_h5ad(spotiphy_dir / "ST_decomposition.h5ad")
    adata_st = sc.read_h5ad(spotiphy_dir / f"{proj}.h5ad")
    adata_decomp = common.add_generated_cell_locations(spotiphy_dir, adata_decomp, adata_st)

    # fresh project metadata (clear Borg state so the loop actually switches projects)
    ProjectMetadata._shared_state.clear()
    ProjectMetadata.create(project_dir, adata_decomp, "cell_type")

    # standard preprocessing + HVG on the ST data
    adata_decomp = common.filter(adata_decomp, min_genes=20, min_locs=20)
    adata_st = common.filter(adata_st, min_genes=20, min_locs=20)
    adata_st.raw = adata_st.copy()
    sc.pp.highly_variable_genes(adata_st, flavor="seurat_v3", n_top_genes=N_TOP_GENES)
    sc.pp.normalize_total(adata_st, target_sum=1e6)
    sc.pp.log1p(adata_st)
    save_orig_st_data(adata_st)
    save_decomposed_st_data(adata_decomp)

    # per-cell-type spatial AnnData + feature selection
    singlecell.make_ct_spatial_adata(adata_decomp)
    singlecell.perform_ct_feature_selection(n_features=N_FEATURES)

    # hypergraph ML dataset
    hgdata = visium.create_hypergraph_data(
        adata_st, n_features=N_FEATURES, num_neighbor_spots=7, n_hyperedge_feats=N_FEATURES)
    hgdata.save()
    print(f"[{proj}] hypergraph saved -> {project_dir}")
print(f"\\nAll {len(TODO)} samples (both sections) prepared through hypergraph creation.")"""))

# build the eval command block at BUILD time from the manifest, but ship it as a STATIC
# markdown cell (nothing to execute — just copy/paste), priority-ordered by spot count.
_flat = sorted(
    [(sec, p, s, n) for sec, (src, man) in SECTIONS_PY.items() for (p, s, n) in man
     if p not in SKIP_PY], key=lambda t: t[3], reverse=True)
_lines = []
for _i, (_sec, _p, _s, _n) in enumerate(_flat, 1):
    _proj = f"Kidney Cancer {_sec} {_p} {_s}"
    _lines.append(f"# {_i}  {_sec} {_p}  ({_n} spots)")
    _lines.append(
        f'CUDA_VISIBLE_DEVICES=0 hyperspot perform-model-evaluation --project "{_proj}" '
        f'--model.model-name CTHGCAEv6Embedder --model.input-dim 3000 --model.embedding-dim 20 '
        f'--model.dropout 0.2 --trainer.num_epochs 1000 --trainer.batch_size 8192 '
        f'--trainer.accelerator cuda --num_runs 10\n')
_cmds_md = "\n".join(_lines)
c2.append(md("""## Model evaluation — one command per sample (priority-ordered)

Model training + 10-seed evaluation is **not run by this notebook**. Run these **individually** in a
terminal (each trains 10 seeds → domains → ct-SVGs; heavy on a shared GPU), ordered by spot count.
PD47171 is omitted — it is already evaluated as the single-donor project.

```bash
""" + _cmds_md + "```"))

nb2 = kernel(new_notebook(cells=c2))
nbf.write(nb2, f"{DEMO}/Human Kidney Cancer Multi-Sample 2 - HyperSpot Pipeline.ipynb")

print("wrote 3 notebooks (both sections):", len(c0), len(c1), len(c2), "cells")
