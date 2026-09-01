# -*- coding: utf-8 -*-
import nbformat
from nbformat.v4 import new_markdown_cell
p = "/workspace/HyperSpot/demo/Human Kidney Cancer Domain-Subtype Correspondence.ipynb"
nb = nbformat.read(p, as_version=4)

CONTEXT = r"""### Dataset context — what this data was built to study

The data is **Li et al. 2022, *Cancer Cell*** ("Mapping single-cell transcriptomes in the intra-tumoral and
associated territories of kidney cancer"): 10x Visium of **renal cell carcinoma (RCC)** with paired tumor
**core** and tumor–normal **interface** sections (our two sections). Its headline biology is **spatial and
interface-driven**: **EMT-high RCC cells enrich at the tumor edge**, and **IL1B⁺ macrophages (Myeloid) at
the tumor–normal interface promote that EMT** (the paper's therapeutic angle). T cells feature mainly as
**CD8⁺ TCR-clonotype** heterogeneity — a *repertoire* phenomenon, not a transcriptomic spatial-domain one —
and T/NK cells are **sparse** in the tissue.

**Why this matters for reading the results below.** The compartments HyperSpot performs best on — **RCC and
Myeloid**, both sections, winning subtype fidelity *and* spatial coherence (Moran's I) — are exactly the
ones carrying this dataset's (spatial, interface) biology. **T-cell** is the laggard on every metric here,
but that is a property of a tumor-focused dataset: T cells are sparse and their signal is clonotypic. For
calibration, on this same data **CELINA excluded T cells and NK cells altogether** (ct-SVGs on 10 of 12 cell
types, both sections), and **STANCE ran all 12 cell types but only the tumor core** — so a weak/absent
T-cell result is consistent with how the ct-SVG methods themselves treated this data."""

# insert after the intro (first markdown cell)
if not any(c.cell_type=="markdown" and c.source.startswith("### Dataset context") for c in nb.cells):
    intro_idx = next(i for i,c in enumerate(nb.cells)
                     if c.cell_type=="markdown" and c.source.startswith("# Human Kidney Cancer"))
    nb.cells.insert(intro_idx+1, new_markdown_cell(CONTEXT))
    ins = 1
else:
    ins = 0

# extend the per-cell-type summary line
n=0
for c in nb.cells:
    if c.cell_type=="markdown" and "the least spatially coherent domains" in c.source and \
       "CELINA excluded T/NK" not in c.source:
        c.source = c.source.replace(
            "Consistent across Interface and Core.",
            "Consistent across Interface and Core. (On this same data **CELINA excluded T/NK cells "
            "entirely**; **STANCE analysed all 12 cell types but only the tumor core** — so the T-cell "
            "difficulty is not unique to HyperSpot.)")
        n+=1
nbformat.write(nb, p)
print(f"inserted context={ins}, patched per-ct line={n}")
