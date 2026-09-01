import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

cells = []
cells.append(md(r"""# Human Kidney Cancer — ct-SVG reproducibility & cross-donor conservation (#6)

Two distinct questions about the stability of HyperSpot's ct-SVGs, both on **MT-free** data:

- **(A) Within-dataset reproducibility** — across the 10 evaluation **seed runs** on one section, how
  reproducible is each cell type's ct-SVG set? (a *method* property.)
- **(B) Cross-donor conservation** — across **patients**, how much do ct-SVGs recur, versus a permutation
  null and versus a tissue-wide-SVG baseline? (a *biological* claim.)

Both are computed by `scripts/conserved_ctsvg.py`; this notebook loads its saved CSVs and renders them.
Neither metric removes housekeeping genes, so both are inflated by ubiquitous ribosomal/OXPHOS genes —
read (A) as *method reproducibility* and (B) as *reproducibility of largely-housekeeping recurrence*, with
the cell-type-specific residual reported separately (see the conserved-marker analysis)."""))

cells.append(md("### Setup"))
cells.append(code(r"""import pandas as pd, numpy as np, matplotlib.pyplot as plt
BASE = "hyperspot_project_files"
ROBUST = ["RCC","EC","Fibro","Epi_non-PT","Myeloid","Epi_PT","B-cell","T-cell"]"""))

cells.append(md("""## (A) Within-dataset ct-SVG stability across 10 seed runs

`mean_jaccard` / `mean_containment` = mean pairwise overlap between runs; `core_frac` = fraction of the
union detected in >= 8/10 runs. High values → the ct-SVG set is a stable method output, not seed-luck."""))
cells.append(code(r"""stab = {sec: pd.read_csv(f"{BASE}/{sec}/ctsvg_run_stability.csv")
        for sec in ["Human Kidney Cancer Core","Human Kidney Cancer Interface"]}
for sec, df in stab.items():
    print(f"\n### {sec}")
    print(df.round(3).to_string(index=False))"""))
cells.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(18,5))
for ax, metric in zip(axes, ["mean_jaccard","mean_containment"]):
    x = np.arange(len(ROBUST)); w=0.38
    for i,(sec,df) in enumerate(stab.items()):
        d = df.set_index("CT").reindex(ROBUST)
        ax.bar(x+(i-0.5)*w, d[metric], w, label=sec.replace("Human Kidney Cancer ",""))
    ax.set_xticks(x); ax.set_xticklabels(ROBUST, rotation=45, ha="right")
    ax.set_ylabel(metric); ax.set_title(f"Run-to-run ct-SVG {metric}"); ax.legend()
fig.suptitle("(A) Within-dataset ct-SVG reproducibility across 10 seeds", y=1.02)
plt.tight_layout(); plt.show()"""))
cells.append(md("""**Read:** abundant compartments (RCC, EC, Fibro, Epi_non-PT) are highly reproducible
(Jaccard ~0.73–0.87, containment ~0.90–0.95, 55–91% core); reproducibility declines with abundance
(Myeloid/Epi_PT moderate, T-cell weak, NK/pDC/Plasma noise) — the same abundance gradient seen throughout."""))

cells.append(md("""## (B) Cross-donor conservation (5 interface donors)

`cons_frac>=2/3` = fraction of the ct-SVG union recurring in ≥2 / ≥3 donors; `enrichment` vs a permutation
null; `tissue_baseline` = the same recurrence for tissue-wide SVGs; `ct_specific_frac` = conserved fraction
after removing tissue-wide SVGs."""))
cells.append(code(r"""cons = pd.read_csv(f"{BASE}/_conserved_ctsvg_interface/conserved_ctsvg_summary.csv")
print(cons.round(3).to_string(index=False))"""))
cells.append(code(r"""d = cons[cons.CT.isin(ROBUST)].set_index("CT").reindex(ROBUST)
fig, ax = plt.subplots(figsize=(11,5)); x=np.arange(len(d))
ax.bar(x-0.28, d["cons_frac>=2"], 0.28, label="ct-SVG conserved (>=2 donors)", color="#2980b9")
ax.bar(x,      d["tissue_baseline>=2"], 0.28, label="tissue-wide-SVG baseline", color="#95a5a6")
ax.bar(x+0.28, d["ct_specific_frac>=2"], 0.28, label="ct-specific (tissue removed)", color="#27ae60")
ax.plot(x, d["null_frac>=2"], "k--o", ms=4, label="permutation null")
ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=45, ha="right")
ax.set_ylabel("conserved fraction (>=2 donors)")
ax.set_title("(B) Cross-donor conservation vs tissue baseline vs null"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()"""))
cells.append(md(r"""**Read (honest, two-sided):** cross-donor conservation is far above the permutation null
(6–24×, p<=0.001) — ct-SVGs *are* patient-reproducible. **But** it sits essentially at the tissue-wide-SVG
baseline (~0.55), and the tissue-removed cell-type-specific fraction collapses to ~0.05–0.15 — so most
cross-patient conservation is generic tissue/housekeeping signal, not cell-type-specific programs. The one
robust cell-type-specific conserved program that survives housekeeping removal is **myeloid antigen
presentation** (MHC / C1Q / CD74 / HLA)."""))

cells.append(md(r"""## Summary

- **(A) HyperSpot's ct-SVGs are a reproducible method output** on a single dataset: abundant cell types
  reach ~0.9 run-to-run containment and >50% ≥8/10 core across 10 seeds.
- **(B) They are also patient-reproducible** (well above null), but this conservation is **largely
  tissue-level/housekeeping**; the cell-type-specific conserved residual is small, with myeloid
  antigen-presentation the clearest genuine signal.
- Both metrics track cell-type **abundance** and are undefined for the sparsest types (Plasma/NK).

*Caveat:* neither (A) nor (B) removes housekeeping genes; both are inflated by ubiquitous ribosomal/OXPHOS
genes and should be read as reproducibility, not biological specificity."""))

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name":"python3","display_name":"Python 3","language":"python"}
nb.metadata["language_info"] = {"name":"python"}
out = "/workspace/HyperSpot/demo/Human Kidney Cancer Conserved ctSVGs.ipynb"
nbf.write(nb, out); print("wrote", out, "with", len(cells), "cells")
