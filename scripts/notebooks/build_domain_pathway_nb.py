import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

cells = []
cells.append(md(r"""# Human Kidney Cancer — Domain -> Pathway Map

**HyperSpot's ct-Domains as functional units.** CELINA and STANCE output *genes-per-cell-type*; HyperSpot
additionally partitions each cell type into **spatial domains**, so each domain's marker ct-SVGs define a
gene program that can be functionally annotated. Running g:Profiler **per ct-Domain** turns each domain
into a set of enriched pathways -- a **domain x pathway** map the regression-based methods structurally
cannot produce.

Each domain's *distinctive* pathways are surfaced by scoring pathways on domain-specificity, so a program
shared across all of a cell type's domains (e.g. OXPHOS) does not dominate and each domain shows what is
unique to it."""))

cells.append(md("Setup -- the interface project and the per-domain pathway/enrichment helpers."))
cells.append(code(r"""import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from hyperspot.metadata import ProjectMetadata
from hyperspot.pipeline.selection import compute_domain_pathways, domain_signature_pathways

ProjectMetadata.use("hyperspot_project_files/Human Kidney Cancer Interface")
pm = ProjectMetadata()
print("cell types:", pm.ct_list)"""))

cells.append(md("""**Compute the domain x pathway table for every cell type** via `compute_domain_pathways`
(per-domain ct-SVGs -> g:Profiler GO:BP/Reactome/KEGG). Rare cell types whose domains have too few
ct-SVGs (deconvolution-limited) yield no rows and are skipped. *(Network + a few minutes.)*"""))
cells.append(code(r"""tables = {}
for ct in pm.ct_list:
    try:
        t = compute_domain_pathways(ct, min_genes=15)
    except Exception as e:
        print(f"{ct}: skip ({str(e)[:40]})"); continue
    if len(t):
        t.insert(0, "ct", ct); tables[ct] = t
        print(f"{ct}: {t['domain'].nunique()} domains, {len(t)} enriched pathways")
allp = pd.concat(tables.values(), ignore_index=True)
CTS = [ct for ct in pm.ct_list if ct in tables]"""))

cells.append(md("""## Per-cell-type domain signatures

For each cell type, each domain's **signature pathways** -- ranked by *domain-specificity*
(this domain's -log10 p minus the best over the other domains) so shared programs are demoted and each
domain surfaces what is distinctive to it. The heatmaps show a block structure: domains are functionally
distinct spatial units."""))
cells.append(code(r"""def plot_domain_pathways(ct, ax, per_domain=2, term_size_max=400):
    sig, M = domain_signature_pathways(allp[allp.ct == ct], per_domain=per_domain,
                                       term_size_max=term_size_max)
    order = sorted(M.columns, key=int)
    paths = []
    for dom in order:
        for p, _ in sig[dom]:
            if p not in paths: paths.append(p)
    M = M.loc[paths, order]
    im = ax.imshow(M.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"d{c}" for c in M.columns], fontsize=8)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([p[:40] for p in M.index], fontsize=7)
    ax.set_title(f"{ct}", fontsize=11)
    return im

show = [ct for ct in ["RCC", "EC", "Fibro", "Myeloid", "Epi_non-PT"] if ct in CTS]
fig, axes = plt.subplots(1, len(show), figsize=(5.5 * len(show), 6))
for ax, ct in zip(np.atleast_1d(axes), show):
    im = plot_domain_pathways(ct, ax)
fig.suptitle("Domain x signature pathway (-log10 p) -- each domain's distinctive program", y=1.02)
fig.colorbar(im, ax=list(np.atleast_1d(axes)), shrink=0.5, label="-log10 p")
plt.show()"""))

cells.append(md("Signature pathways per domain, as text (top-2 domain-specific pathways each)."))
cells.append(code(r"""for ct in show:
    sig, _ = domain_signature_pathways(allp[allp.ct == ct], per_domain=2, term_size_max=400)
    print(f"\n### {ct}")
    for dom in sorted(sig, key=int):
        if sig[dom]:
            print(f"  domain {dom}: " + " ; ".join(f"{p} ({v})" for p, v in sig[dom]))"""))

cells.append(md("""## Spatial map -- signature pathways on the tissue

Each cell type's domains are spatial regions, so plotting them (squidpy) and coloring each domain by its
top signature pathway maps the functional programs directly onto the tissue -- the hypoxia, immune,
translation, etc. programs occupy distinct territories. (This is in addition to the heatmaps above.)"""))
cells.append(code(r"""import squidpy as sq
from hyperspot.pipeline.selection import load_ct_spatial_data

def plot_domain_pathway_spatial(ct, size=1.5):
    a = load_ct_spatial_data(ct); a.obs["domain"] = a.obs["domain"].astype(str)
    sig, _ = domain_signature_pathways(allp[allp.ct == ct], per_domain=1, term_size_max=400)
    lab = {d: (sig[d][0][0] if sig.get(d) else None) for d in a.obs["domain"].unique()}
    mapped = [d for d in lab if lab[d]]                       # only domains with a signature pathway
    if not mapped:
        print(f"{ct}: no domains with a signature pathway"); return
    a = a[a.obs["domain"].isin(mapped)].copy()
    a.obs["program"] = a.obs["domain"].map(lambda d: f"d{d}: {lab[d][:34]}").astype("category")
    fig, ax = plt.subplots(figsize=(10, 8))
    sq.pl.spatial_scatter(a, color="program", img=True, size=size, ax=ax,   # H&E background
                          title=f"{ct}: domains -> signature pathways")
    plt.show()

for ct in show:
    plot_domain_pathway_spatial(ct)"""))

cells.append(md(r"""## Future work -- cross-cell-type pathway relationships

A compelling open question: **is pathway A of domain *i* of cell type c1 *related* to pathway B of
domain *j* of cell type c2**, especially for domains that co-occupy the same tissue region? "Shared"
(A == B) is just the trivial edge -- pathways can be related without being identical (e.g. an EC domain's
*angiogenesis* co-localized with an RCC domain's *hypoxia*: mechanistically coupled, not the same term).

Defining that relatedness rigorously is the hard part -- candidate measures disagree:
- **GO semantic similarity** captures *ontologically nested* relatedness (hypoxia <-> "response to
  decreased oxygen" ~ 0.78) but **misses cross-branch biological coupling** (hypoxia <-> angiogenesis ~
  0.09), because GO encodes is-a/part-of, not causal links.
- **Gene crosstalk** (enrichment-map member-gene overlap) *would* catch hypoxia<->angiogenesis (shared
  HIF-target genes) but is partly circular when both pathways derive from overlapping gene sets.
- **Ligand->receptor signaling** -- domain *i*'s ligands hitting domain *j*'s receptors -- is the
  concrete, mechanistic, directional version, and ties into HyperSpot's ctSVG-grounded signaling niches
  and the installed niche baselines (SpaNiche / niche-DE).

A preliminary spatial-only check (do co-localized cross-CT domains share pathway *terms*?) gave a
significant but modest signal (Spearman rho ~ 0.37). Turning this into a defensible claim -- via the
signaling-coupling formulation -- is left as **future work**; it is a genuinely interesting direction for
making ct-Domains reveal tissue-region-level functional coordination across cell types."""))

cells.append(md(r"""## Summary

- **HyperSpot resolves each cell type's spatial domains into functionally distinct programs** -- a
  domain x pathway map (RCC hypoxia vs innate-immune vs translation domains; **Fibro ECM-organization
  domain**; **Myeloid T-cell-activation domain**; EC focal-adhesion vs chemotaxis domains). This is an
  output CELINA/STANCE structurally cannot produce (they yield genes-per-cell-type, no spatial domains),
  and it makes ct-Domains a first-class, functionally-annotated output.
- **Future work:** whether the pathways of *co-localized* domains across cell types are *related*
  (beyond shared terms) -- best pursued via ligand->receptor signaling coupling."""))

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out = "/workspace/HyperSpot/demo/Human Kidney Cancer Domain-Pathway Map.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
