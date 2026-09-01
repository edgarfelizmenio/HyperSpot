# -*- coding: utf-8 -*-
import nbformat
p = "/workspace/HyperSpot/demo/Human Kidney Cancer Domain-Subtype Correspondence.ipynb"
nb = nbformat.read(p, as_version=4)

CAVEAT = r"""

> **Caveat — this metric is expression-defined end to end.** The `annotation` subtypes are scRNA
> *expression* clusters (marker-named; the reference carries only `X_pca`/`X_umap`, no spatial modality),
> and `sc.tl.ingest` transfers them by kNN in the reference's **expression** PCA space. So a cell's target
> label is set by its expression neighbours, and PCA+Leiden clusters those *same* cells in that *same* space
> — it aligns with the labels almost by construction, whereas HyperSpot's spatial term makes it *deviate*.
> This comparison is therefore a **cell-state fidelity check** (do the ST clusters recover known
> expression-defined states?), **not** a spatial-vs-expression ranking — the ground truth has no spatial
> component with which to reward a spatial method."""

MORAN = r"""## Spatial coherence of the partitions — Moran's I

**What Moran's I measures.** Moran's I is the standard statistic for *spatial autocorrelation*: given a
value at every cell and a spatial-neighbour graph, it asks whether neighbouring cells tend to carry similar
values. `I ≈ +1` → neighbours are alike (a contiguous spatial pattern); `I ≈ 0` → no spatial structure
(values placed at random); `I < 0` → neighbours differ (checkerboard). Formally,
`I = (N / S0) · [ Σ_i Σ_j w_ij (x_i − xbar)(x_j − xbar) ] / [ Σ_i (x_i − xbar)^2 ]`, where `w_ij = 1` if cell
*j* is a spatial neighbour of *i* (here, one of its 6 nearest cells) and 0 otherwise, `S0 = Σ_i Σ_j w_ij`,
and `xbar` is the mean of `x`.

**"Moran's I of a domain."** A domain is a categorical label, not a number, so we turn each domain *d* into
a 0/1 **indicator** — `x_i = 1` if cell *i* belongs to domain *d*, else `0` — and compute Moran's I of that
indicator. It is **high when the cells of domain *d* sit together in space** (a contiguous territory) and
**≈0 when they are scattered** across the tissue. Reporting the **mean Moran's I over all of a method's
domains** gives that partition's overall spatial coherence (one number per method, per cell type).

**How it's computed here.** For each method: one-hot the cluster labels (cells × clusters), attach the
cells' `spatial` coordinates, build a 6-nearest-neighbour spatial graph
(`squidpy.gr.spatial_neighbors`), run `squidpy.gr.spatial_autocorr(mode="moran")` to get Moran's I for each
cluster indicator, and average over clusters.

**An important, non-obvious result — read the numbers, not the intuition.** *All three* methods score high
(~0.4–0.9), **including PCA+Leiden**, because deconvolution imputes **spatially smooth (spot-level)
expression**, so even a purely expression-based clustering inherits spatial structure. The common intuition
that "expression clusters are spatially scattered" is **false on deconvolved data.** HyperSpot is more
coherent than Leiden for the **abundant compartments** (RCC, Myeloid; both sections, by +0.05–0.12) but
**less** coherent for the sparse **T-cell** (few cells → noisier domains). So HyperSpot's spatial-coherence
edge is **real for the tumor/abundant cell types but modest and not universal** — not a decisive separation.

**Consequence for ct-SVGs (honest version).** Because Leiden's clusters here already carry spatial signal,
"cluster with Leiden then run DE" would yield *partly* spatial markers — so the case for HyperSpot rests
**not** on "Leiden isn't spatial", but on (i) its modestly higher coherence in the tumor compartment,
(ii) an explicit spatial model with a proper spatial null rather than circular cluster-DE, and (iii) the
independent ct-SVG validations (CELINA/STANCE agreement, TCGA survival)."""

SUMMARY = r"""## Summary — the honest, two-sided result

**Subtype recovery (expression fidelity).** At matched resolution (both at Leiden res=1.0 on the same
deduplicated cells), **PCA+Leiden+dedup is slightly but consistently *ahead* of HyperSpot** on V-measure and
ARI in all six section×cell-type cases, usually with *fewer* clusters — and both **far exceed scMiner**.

**But mind what the metric is.** The benchmark is **expression-defined end to end**: the `annotation`
subtypes are scRNA *expression* clusters, and `sc.tl.ingest` assigns each ST cell its label by kNN in the
reference's *expression* PCA space. PCA+Leiden then clusters those same cells in that same space, so it
matches the labels almost by construction; HyperSpot's spatial term makes it deviate and score slightly
lower. So this is a **cell-state fidelity check**, **not** a spatial-vs-expression ranking — "Leiden beats
HyperSpot" here is largely a property of an expression-only ground truth, and the test cannot rank a spatial
method above a pure-expression one in either direction. The defensible reading is:
**HyperSpot's domains are as cell-state-faithful as dedicated expression clustering (≫ scMiner).**

**Spatial coherence — real but modest, and mixed.** By Moran's I, HyperSpot's domains are more spatially
contiguous than the expression clusterings for the **abundant** cell types (RCC, Myeloid; both sections) but
**not** for sparse T-cell. Crucially, *all* methods are fairly coherent here because the imputed decomposed
expression is spatially smooth — so this is a modest edge in the tumor compartment, not a clean separation,
and the earlier intuition that expression clusters are spatially random does **not** hold on deconvolved data.

**Net positioning.** On this validation HyperSpot neither wins subtype recovery (Leiden does, on an
expression-only metric) nor decisively wins spatial coherence (only for abundant cell types). What it
delivers is domains that are **simultaneously cell-state-faithful and spatially organised**, backing an
*explicit* spatial model for ct-SVG detection (with a spatial null, avoiding circular cluster-DE) — while
its distinctive, better-supported advantages live in the ct-SVG results themselves (CELINA/STANCE agreement,
TCGA survival, domain→pathway maps), not in out-clustering expression baselines.

**Per cell type:** RCC strongest and most stable; Myeloid moderate; T-cell weakest and deconvolution-limited
(few cells, many fine states, and — consistent with the Moran's I — the least spatially coherent domains).
Consistent across Interface and Core.

*Caveats:* the ground truth and its transfer are expression-only (above); transfer onto imputed per-cell
expression is soft; abundant subtypes dominate; metrics are moderate; scMiner's spread is from 10 unseeded
MICA repeats (no seed flag → descriptive, not reproducible)."""

n = 0
for c in nb.cells:
    if c.cell_type != "markdown":
        continue
    s = c.source
    if s.startswith("## Four-method comparison") and "expression-defined end to end" not in s:
        c.source = s.rstrip() + CAVEAT; n += 1
    elif s.startswith("## Spatial coherence of the partitions"):
        c.source = MORAN; n += 1
    elif s.startswith("## Summary"):
        c.source = SUMMARY; n += 1
nbformat.write(nb, p)
print("patched", n, "markdown cells")
