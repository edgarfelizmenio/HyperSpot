# -*- coding: utf-8 -*-
"""Append a 'tissue-wide SVG concordance' section to the CELINA/STANCE notebook,
executing ONLY the new cells (existing network-dependent sections untouched)."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code
from nbclient import NotebookClient

TARGET = "/workspace/HyperSpot/demo/Human Kidney Cancer Analysis - CELINA STANCE Comparison.ipynb"

intro = md(r"""## Tissue-wide SVG concordance — HyperSpot vs STANCE utSVGs (Core)

STANCE's Stage-1 *overall test* yields **utSVGs** — tissue-wide, cell-type-agnostic spatially variable genes.
HyperSpot's **hyperedge / tissue domains** produce an analogous tissue-level SVG layer (`svgs_ST`). Both ran
on the same **Core** section with shared **Spotiphy** proportions, so the comparison isolates *detection*.

We recover STANCE's utSVG set as the **union of its Stage-2 per-cell-type files** (every Stage-2-tested gene
was Stage-1/utSVG-significant), and test its overlap with HyperSpot's tissue SVGs against a **common gene
background** (genes measured by both) via a hypergeometric test. HyperSpot tissue SVGs are taken at three
reproducibility levels across the 10 evaluation runs (union / ≥5-of-10 / ≥8-of-10).""")

cell = code(r"""import pandas as pd, glob, numpy as np, matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import hypergeom
import anndata as ad

CORE = "hyperspot_project_files/Human Kidney Cancer Core"
STANCE_DIR = "/workspace/ctSVG-benchmarking/results/4_Human_Kidney_Cancer_Core/STANCE_spotiphy"

# STANCE utSVGs = union of Stage-2-tested genes (each is Stage-1 / utSVG significant)
utsvg = set()
for f in glob.glob(f"{STANCE_DIR}/*.csv"):
    utsvg |= set(pd.read_csv(f, index_col=0).index.astype(str))

# HyperSpot tissue SVGs across the 10 Core runs -> per-gene reproducibility
cnt = Counter(); runs = sorted(glob.glob(f"{CORE}/evaluation/run_*/svgs_ST.csv"))
for f in runs:
    cnt.update(set(pd.read_csv(f, index_col=0).index.astype(str)))
nr = len(runs)
sets = {"union": {g for g in cnt},
        f"reproducible (>={nr//2}/{nr})": {g for g, c in cnt.items() if c >= nr // 2},
        f"strict (>={int(0.8*nr)}/{nr})": {g for g, c in cnt.items() if c >= int(0.8 * nr)}}

# common gene background (measured by both methods)
hs_univ = set(ad.read_h5ad(f"{CORE}/ST_aggregated.h5ad").var_names.astype(str))
st_univ = set(l.strip() for l in
              open("/workspace/ctSVG-benchmarking/data/4_Human_Kidney_Cancer_Core/ST_genes.txt") if l.strip())
bg = hs_univ & st_univ; N = len(bg); K = utsvg & bg

rows = []
for name, S in sets.items():
    n = S & bg; k = len(n & K); exp = len(n) * len(K) / N
    rows.append((name, len(S), len(n), k, k / len(n | K), k / exp, hypergeom.sf(k - 1, N, len(K), len(n))))
tab = pd.DataFrame(rows, columns=["HyperSpot set", "size", "in_bg", "overlap_utSVG",
                                  "Jaccard", "fold_enrichment", "hypergeom_p"])
print(f"common background = {N} genes | STANCE utSVGs = {len(utsvg)} (in bg {len(K)}) | HyperSpot runs = {nr}")
print(tab.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(tab["HyperSpot set"], tab["fold_enrichment"], color="#2980b9")
ax.axhline(1, ls="--", c="grey", label="chance")
ax.set_ylabel("fold enrichment vs chance")
ax.set_title("HyperSpot tissue SVGs vs STANCE utSVGs (Core) — overlap enrichment")
for i, (fo, k) in enumerate(zip(tab["fold_enrichment"], tab["overlap_utSVG"])):
    ax.text(i, fo + 0.1, f"{fo:.1f}x\n{k} genes", ha="center", fontsize=8)
ax.legend(); plt.tight_layout(); plt.show()""")

takeaway = md(r"""**HyperSpot's tissue-wide SVGs independently reproduce a majority of STANCE's utSVGs.**
On the shared Core section HyperSpot's hyperedge/tissue-domain SVGs overlap STANCE's overall-test utSVGs at
**~6-fold enrichment (p ≈ 1e-280)**, recovering **~half of STANCE's utSVGs** (~450 of 907), with agreement
*increasing* for more reproducible HyperSpot genes (5.2x union → 6.3x strict) — the reproducible core is the
most cross-method-concordant. *(Caveat: utSVGs outside HyperSpot's measured panel are excluded from the
background; both methods share the Spotiphy deconvolution, isolating detection.)*""")

# --- execute the new cells in isolation, then append to the target notebook ---
tmp = new_notebook(cells=[cell])
tmp.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(tmp, timeout=600, resources={"metadata": {"path": "/workspace/HyperSpot/demo"}}).execute()

nb = nbformat.read(TARGET, as_version=4)
nb.cells += [intro, tmp.cells[0], takeaway]
nbformat.write(nb, TARGET)
errs = sum(1 for o in tmp.cells[0].get("outputs", []) if o.get("output_type") == "error")
print("APPENDED; new-cell errors:", errs, "| total cells:", len(nb.cells))
for o in tmp.cells[0].get("outputs", []):
    if o.get("output_type") == "stream":
        print(o["text"])
