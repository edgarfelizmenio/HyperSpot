# HyperSpot — Methods

## Biological Target

HyperSpot detects **per-cell-type spatial domain marker SVGs** — genes that distinguish spatial domains within a cell type. This is a biologically distinct subcategory of ctSVGs:

- **ctSVG** (oracle definition): any gene with spatial autocorrelation within a cell type (Moran's I / R² based)
- **Domain marker ctSVG** (HyperSpot): a gene that marks a specific spatial domain within a cell type (Wilcoxon DE between domains)

A smooth gradient gene has high Moran's I but does not cleanly separate spatial domains. A niche-specific gene may be a strong domain marker with moderate global Moran's I. Domain marker ctSVGs are a proper subset of ctSVGs. The Wilcoxon scoring in HyperSpot correctly tests the domain marker hypothesis — replacing it with Moran's I would make HyperSpot redundant with CELINA/STANCE.

The apparent oracle–method mismatch (high FDR against the Moran's I oracle) reflects this difference in biological definition, not a calibration failure. A better oracle for HyperSpot is the **spot co-membership Moran's I** (see ctsvg-benchmark-sim METHODS/roadmap).

---

## ctSVG Scoring for Benchmarking Evaluation

This section describes `compute_ct_svg_scores` in `hyperspot/pipeline/results.py`, which produces per-(gene, cell type) scores for comparison against CELINA, STANCE, and other ctSVG detection methods using the standard benchmarking evaluation metrics (EP, AUPRC, FDP, Power, KS distance, Type I error).

### Background

HyperSpot identifies ctSVGs by clustering cells of a given cell type into spatial domains (via hypergraph-based embedding and Leiden clustering), then running differential expression analysis across those domains. The primary output is a filtered, domain-level gene list per cell type — genes that are significantly upregulated in at least one spatial domain compared to all other domains of the same cell type.

This domain-level structure differs from methods like CELINA and STANCE, which output a single p-value per (gene, cell type) pair. A direct evaluation comparison therefore requires aggregating HyperSpot's domain-level results into a single score per (gene, cell type).

### Biological Rationale

A gene is considered a **ctSVG for a given cell type** if it shows high spatial specificity in at least one spatial domain of that cell type — i.e., it is significantly more expressed in a subset of the tissue where that cell type is concentrated compared to other spatial regions of the same cell type. The domain with the strongest differential expression signal represents the best evidence of spatial variability for that gene.

### Method

For each cell type:

1. **Wilcoxon rank-sum test (1-vs-all per domain)**
   - `sc.tl.rank_genes_groups` is run on the cell-type-specific AnnData (`adata_ct`) grouped by spatial domain label (`obs['domain']`)
   - Each domain is tested against all other domains of the same cell type (not the full tissue)
   - This produces a Wilcoxon score and raw p-value for every (gene, domain) pair, across all genes and all domains
   - No filtering is applied at this stage

2. **Domain aggregation — max score selection**
   - For each gene, the domain with the highest Wilcoxon score is selected
   - The raw p-value from that best domain is retained as the gene's representative p-value
   - This answers the question: *does this gene show spatial specificity in at least one domain of this cell type?*
   - This is analogous to CELINA's Cauchy combination across spatial kernels

   **Why max score rather than Cauchy combination?**
   The 1-vs-all Wilcoxon tests across domains are not independent — each domain's "rest" group includes cells from all other domains, creating positive correlation between domain p-values. Cauchy combination assumes independence and is therefore not appropriate. Max score (equivalent to min p-value selection) is the correct aggregation under this test structure.

3. **Multiple testing correction**
   - Benjamini–Hochberg (BH) FDR correction is applied across all genes **per cell type**, matching the convention used by CELINA and STANCE
   - Genes not present in any domain's test results receive `pval = NaN` and `adj_pval = 1.0` (treated as untested)

4. **Output format**
   - One CSV per cell type, with columns `CombinedPvals` (raw p-value from best domain) and `adj_pval` (BH-corrected)
   - The Wilcoxon score from the best domain serves as the ranking signal for EP and AUPRC computation (higher score = more significant)
   - Format matches the evaluation notebook's `load_run` expectations

### Evaluation Metric Availability

| Metric | Available | Notes |
|--------|-----------|-------|
| Early Precision (EP) | ✅ | Ranked by Wilcoxon score |
| AUPRC | ✅ | Ranked by Wilcoxon score |
| FDP | ✅ | `adj_pval < 0.05` threshold |
| Power | ✅ | `adj_pval < 0.05` threshold |
| KS distance | ✅ | Raw p-values available for null genes |
| Type I error | ✅ | Raw p-values available for null genes |

All six standard benchmarking metrics are computable. This is in contrast to HyperSpot's filtered pipeline output (which retains only significant, upregulated, well-expressed genes), which would preclude KS distance and Type I error computation.

### Usage

```python
from hyperspot.pipeline.results import compute_ct_svg_scores

compute_ct_svg_scores(
    out_dir="results/1_Human_Breast_Cancer/HyperSpot",
    key="domain",  # obs column containing spatial domain labels
)
```

Run this after `compute_celltype_spatial_domains` has assigned domain labels to each cell type's AnnData. The output directory can then be added to the `RUNS` dict in the evaluation notebook.
