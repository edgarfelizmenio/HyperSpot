# Notebook builders & patchers

Scripts that generate / modify the analysis notebooks in `demo/`. The executed `.ipynb` files are the
canonical artifacts; these are their source (run them to regenerate or maintain the notebooks).

| script | produces / modifies | notes |
|---|---|---|
| `build_subtype_nb.py` | `Human Kidney Cancer Domain-Subtype Correspondence.ipynb` | Source of truth; kept in sync with the notebook's markdown. Builds the 4-method comparison (HyperSpot/Leiden/scMiner), subtype ARI, partition Moran's I, 3-method robustness, 4-panel spatial. |
| `build_kidney_analysis_nb.py` | `Human Kidney Cancer Analysis - CELINA STANCE Comparison.ipynb` | Sections A–F (cross-section consistency, metagenes, known-gene DB overlap, 3-way Venn, TCGA survival). |
| `add_concordance_section.py` | appends to the CELINA/STANCE Comparison notebook | Adds the tissue-wide SVG concordance section (#19: HyperSpot tissue SVGs vs STANCE utSVGs); executes only the new cells. |
| `build_domain_pathway_nb.py` | `Human Kidney Cancer Domain-Pathway Map.ipynb` | Per-ct-Domain g:Profiler enrichment + spatial pathway maps (#17). |
| `build_multisample_nbs.py` | `Human Kidney Cancer Multi-Sample {0,1,2}.ipynb` | Per-donor data prep / Spotiphy / pipeline for the 9 donor samples. |
| `patch_mt_filter.py` | the three `* 2 - HyperSpot Pipeline.ipynb` | Wires the ENSG-based mitochondrial-gene filter (`is_mt_gene`) into `common.filter`. |
| `patch_md.py`, `patch_context.py` | Domain-Subtype Correspondence notebook | In-place markdown corrections (honest Moran's-I framing, dataset-context note, expression-only-metric caveat). Applied after `build_subtype_nb.py`; the builder was also synced to match. |

## Regeneration order (Domain-Subtype notebook)
`build_subtype_nb.py` → (already-synced markdown; `patch_md.py` / `patch_context.py` were the incremental
edits, now folded into the builder). Re-running the builder alone reproduces the current notebook structure;
execute it in the `HyperSpot` conda env from `demo/`.

## Regeneration order (CELINA/STANCE Comparison notebook)
`build_kidney_analysis_nb.py` → `add_concordance_section.py`.
