"""
conserved_ctsvg.py — cross-donor conservation of HyperSpot ct-SVGs.

For a set of donor projects, each with 10 eval runs of per-cell-type ct-SVGs
(evaluation/run_*/svgs_ct_<ct>.csv) and tissue-wide SVGs (svgs_ST.csv), quantify how
much a cell type's ct-SVGs RECUR ACROSS PATIENTS, against:
  * a permutation NULL (random gene sets of matched size from the common panel), and
  * a tissue-wide-SVG BASELINE (do cell-type-specific genes conserve beyond generic
    spatial genes? — and a cell-type-specific conserved fraction with tissue SVGs removed).

Per-donor ct-SVG set = genes detected in >= `min_run_frac` of that donor's runs (stable set).
"""
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd
import anndata as ad


def _stable_set(donor_dir, fname, min_run_frac):
    runs = sorted(glob.glob(f"{donor_dir}/evaluation/run_*/{fname}"))
    if not runs:
        return set(), 0
    c = Counter()
    for f in runs:
        try:
            c.update(set(pd.read_csv(f, index_col=0).index.astype(str)))
        except Exception:
            pass
    thr = max(1, int(np.ceil(min_run_frac * len(runs))))
    return {g for g, n in c.items() if n >= thr}, len(runs)


def _conserved_fraction(sets, min_donors):
    rec = Counter()
    for s in sets:
        rec.update(s)
    union = set(rec)
    if not union:
        return 0.0, set(), 0
    conserved = {g for g, n in rec.items() if n >= min_donors}
    return len(conserved) / len(union), conserved, len(union)


def _null_fractions(sizes, n_bg, min_donors, n_perm, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    for b in range(n_perm):
        rec = Counter()
        for s in sizes:
            rec.update(rng.choice(n_bg, size=min(s, n_bg), replace=False).tolist())
        union = len(rec)
        out[b] = (sum(1 for n in rec.values() if n >= min_donors) / union) if union else 0.0
    return out


def conserved_ctsvg_fraction(donor_dirs, cell_types, min_run_frac=0.5, min_donors=2,
                             n_perm=1000, seed=0, background=None, out_dir=None):
    # common gene panel across donors (background for the null)
    if background is None:
        panels = []
        for d in donor_dirs:
            p = f"{d}/ST_aggregated.h5ad"
            if os.path.exists(p):
                panels.append(set(ad.read_h5ad(p).var_names.astype(str)))
        background = set.intersection(*panels) if panels else set()
    n_bg = len(background)

    rows, conserved_genes = [], {}
    for ct in cell_types:
        sets, nruns = [], []
        for d in donor_dirs:
            s, nr = _stable_set(d, f"svgs_ct_{ct}.csv", min_run_frac)
            sets.append(s & background); nruns.append(nr)
        nz = [len(s) for s in sets if len(s) > 0]
        n_donors = len(nz)
        if n_donors < 2:
            rows.append((ct, n_donors, np.median(nz) if nz else 0, 0, 0.0, 0.0,
                         np.nan, np.nan, np.nan, 0.0, 0.0)); continue

        frac2, cons2, union = _conserved_fraction(sets, min_donors)
        frac3, cons3, _ = _conserved_fraction(sets, 3)
        sizes = [len(s) for s in sets if len(s) > 0]
        null = _null_fractions(sizes, n_bg, min_donors, n_perm, seed)
        null_m, null_s = null.mean(), null.std()
        enrich = frac2 / null_m if null_m > 0 else np.inf
        emp_p = (np.sum(null >= frac2) + 1) / (n_perm + 1)

        # tissue baseline + cell-type-specific (tissue-removed) conserved fraction
        tsets = [_stable_set(d, "svgs_ST.csv", min_run_frac)[0] & background for d in donor_dirs]
        tfrac2, tcons, _ = _conserved_fraction(tsets, min_donors)
        tissue_union = set().union(*tsets) if tsets else set()
        sets_ct = [s - tissue_union for s in sets]
        ctfrac2, cons2_ct, _ = _conserved_fraction(sets_ct, min_donors)

        rows.append((ct, n_donors, int(np.median(sizes)), union, frac2, frac3,
                     null_m, enrich, emp_p, tfrac2, ctfrac2))
        conserved_genes[ct] = {"conserved_ge2": sorted(cons2), "conserved_ge3": sorted(cons3),
                               "conserved_ct_specific_ge2": sorted(cons2_ct)}

    tab = pd.DataFrame(rows, columns=[
        "CT", "n_donors", "median_size", "union", "cons_frac>=2", "cons_frac>=3",
        "null_frac>=2", "enrichment", "emp_p", "tissue_baseline>=2", "ct_specific_frac>=2"])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tab.to_csv(f"{out_dir}/conserved_ctsvg_summary.csv", index=False)
        for ct, g in conserved_genes.items():
            pd.Series(g["conserved_ge3"]).to_csv(
                f"{out_dir}/conserved_{ct.replace('-','_')}_ge3.csv", index=False, header=["gene"])
    return tab, conserved_genes


def run_stability(section_dir, cell_types, core_frac=0.8):
    """Within-dataset ct-SVG stability across the 10 evaluation SEED runs (distinct
    from cross-donor conservation): for each cell type, mean pairwise Jaccard and
    containment between runs, plus the fraction forming a >=core_frac 'core'.
    This measures METHOD reproducibility on one dataset, not biological conservation
    across patients — and (like the cross-donor fraction) is inflated by housekeeping
    genes, so report it as a reproducibility check, not a biology claim."""
    from itertools import combinations
    runs = sorted(glob.glob(f"{section_dir}/evaluation/run_*"))
    rows = []
    for ct in cell_types:
        sets = []
        for r in runs:
            f = f"{r}/svgs_ct_{ct}.csv"
            if os.path.exists(f):
                d = pd.read_csv(f)
                s = set(d.iloc[:, 0].astype(str)) if len(d) else set()
                if s:
                    sets.append(s)
        if len(sets) < 2:
            rows.append((ct, len(sets), 0, 0, np.nan, np.nan, np.nan)); continue
        cnt = Counter()
        for s in sets:
            cnt.update(s)
        union = set(cnt); core = {g for g, c in cnt.items() if c >= int(core_frac * len(sets))}
        jac = np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)])
        con = np.mean([len(a & b) / min(len(a), len(b)) for a, b in combinations(sets, 2)])
        rows.append((ct, len(sets), int(np.mean([len(s) for s in sets])), len(union),
                     len(core) / len(union), jac, con))
    return pd.DataFrame(rows, columns=["CT", "n_runs", "mean_n", "union",
                                       f"core_frac(>={core_frac:.0%})", "mean_jaccard", "mean_containment"])


if __name__ == "__main__":
    base = "/workspace/HyperSpot/demo/hyperspot_project_files"
    cts = ["RCC", "EC", "Fibro", "Epi_non-PT", "Myeloid", "Epi_PT", "B-cell", "T-cell",
           "NK", "pDC", "Mast", "Plasma"]
    pd.set_option("display.width", 170)

    # (A) within-dataset seed stability — both pooled sections (method reproducibility)
    print("=" * 70, "\n(A) WITHIN-DATASET ct-SVG stability across 10 seed runs\n", "=" * 70)
    for sec in ["Human Kidney Cancer Core", "Human Kidney Cancer Interface"]:
        st = run_stability(f"{base}/{sec}", cts)
        st.to_csv(f"{base}/{sec}/ctsvg_run_stability.csv", index=False)
        print(f"\n### {sec}\n" + st.round(3).to_string(index=False))

    # (B) cross-donor conservation — interface donors (biological conservation)
    print("\n" + "=" * 70, "\n(B) CROSS-DONOR ct-SVG conservation (interface donors)\n", "=" * 70)
    donors = sorted(glob.glob(f"{base}/Kidney Cancer Interface PD*"))
    print(f"Interface donors ({len(donors)}):")
    for d in donors:
        print("  ", os.path.basename(d))
    tab, genes = conserved_ctsvg_fraction(
        donors, cts, min_run_frac=0.5, min_donors=2, n_perm=1000,
        out_dir=f"{base}/_conserved_ctsvg_interface")
    print("\n" + tab.round(3).to_string(index=False))
    print("\nConserved (>=3 donors) counts:",
          {ct: len(g["conserved_ge3"]) for ct, g in genes.items()})
