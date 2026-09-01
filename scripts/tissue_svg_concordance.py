"""
tissue_svg_concordance.py — HyperSpot tissue SVGs vs STANCE utSVGs (task #19), reproducible.

Every number reported for #19 comes from here. Run in the HyperSpot conda env from `demo/`:
    conda run -n HyperSpot python ../scripts/tissue_svg_concordance.py

Inputs (all on disk):
  - HyperSpot tissue SVGs : <proj>/evaluation/run_*/svgs_ST.csv          (10 MT-free runs)
  - STANCE utSVGs         : ctSVG-benchmarking/.../STANCE_spotiphy/utSVGs_stage1.csv
                            (Stage-1 BY<0.05; produced by run_stance_stage1.R, MT-free)
  - tissue expression     : <proj>/ST_aggregated.h5ad  (MT-free, spatial coords)
  - common gene panel     : HyperSpot ST panel ∩ STANCE ST_genes.txt

Sections:
  A. containment + hypergeometric enrichment of HyperSpot tissue SVGs in STANCE utSVGs
  B. Moran's I of each gene group (HyperSpot / STANCE-only / background)
  C. is the Moran's I gap a spatial-method effect or the DE filter? -> expression control + filter sweep
  D. recall of STANCE-only genes as HyperSpot's pts/logfc filter is dropped, and HyperSpot DE scores
  E. HyperSpot-only genes (matched significance) — does HyperSpot flag genes STANCE misses?
"""
import glob
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import squidpy as sq
from scipy.stats import hypergeom

from hyperspot.metadata import ProjectMetadata
from hyperspot.pipeline.selection import _compute_spatial_de_table
from hyperspot.utils import load_aggr_st_data

DEMO   = "/workspace/HyperSpot/demo"
PROJ   = f"{DEMO}/hyperspot_project_files/Human Kidney Cancer Core"
STANCE = "/workspace/ctSVG-benchmarking/results/4_Human_Kidney_Cancer_Core/STANCE_spotiphy/utSVGs_stage1.csv"
STGENES= "/workspace/ctSVG-benchmarking/data/4_Human_Kidney_Cancer_Core/ST_genes.txt"
MT = {"ENSG00000198886","ENSG00000198888","ENSG00000228253","ENSG00000198899","ENSG00000198695",
      "ENSG00000198840","ENSG00000198938","ENSG00000198804","ENSG00000198712","ENSG00000212907",
      "ENSG00000198763","ENSG00000198727","ENSG00000198786"}


def load():
    ProjectMetadata.use(PROJ)
    aggr = load_aggr_st_data()
    # unfiltered per-gene tissue DE table (best domain per gene): scores/pts/logfc/pvals_adj
    flat = _compute_spatial_de_table(aggr, key="domain", method="wilcoxon", min_pts=0, min_logfc=-1e9,
                                     require_specificity=False, pval_adjust="scanpy",
                                     filter_mode="none", best_domain_per_gene=True).set_index("gene")
    # tissue-level Moran's I + mean expression on the same matrix
    sq.gr.spatial_neighbors(aggr, coord_type="generic", n_neighs=6)
    sq.gr.spatial_autocorr(aggr, mode="moran", genes=list(aggr.var_names), n_perms=None, seed=0)
    moran = aggr.uns["moranI"]["I"]
    X = aggr.X.toarray() if sp.issparse(aggr.X) else np.asarray(aggr.X)
    expr = pd.Series(X.mean(0), index=aggr.var_names)
    m = set(moran.index)
    ut = set(pd.read_csv(STANCE)["gene"].astype(str)) & m
    cnt = Counter()
    for f in sorted(glob.glob(f"{PROJ}/evaluation/run_*/svgs_ST.csv")):
        cnt.update(set(pd.read_csv(f, index_col=0).index.astype(str)))
    hs_default = {g for g, c in cnt.items() if c >= 5} & m
    return aggr, flat, moran, expr, m, ut, hs_default, cnt


def sec_A(m, ut, cnt):
    print("\n== A. containment + hypergeometric enrichment ==")
    hs_univ = m
    st_univ = {l.strip() for l in open(STGENES) if l.strip()}
    bg = hs_univ & st_univ; N = len(bg); K = ut & bg
    print(f"  background(panel)={N}  STANCE utSVGs in bg={len(K)}")
    for name, thr in [("union", 1), (">=5/10", 5), (">=8/10", 8)]:
        S = {g for g, c in cnt.items() if c >= thr} & bg
        k = len(S & K); exp = len(S) * len(K) / N
        print(f"  {name:8s} nHS={len(S):5d} overlap={k:5d} ({100*k/len(S):.0f}%) "
              f"Jaccard={k/len(S|K):.3f} fold={k/exp:.2f} hyperg_p={hypergeom.sf(k-1,N,len(K),len(S)):.1e}")


def sec_B(moran, ut, hs_default, m):
    print("\n== B. Moran's I by group ==")
    both = hs_default & ut; so = ut - hs_default; nei = m - ut - hs_default
    for name, s in [("HyperSpot SVGs", both), ("STANCE-only", so), ("background", nei)]:
        v = moran.loc[list(s)].dropna()
        print(f"  {name:16s} n={len(v):5d} median I={v.median():.3f} mean={v.mean():.3f} "
              f"%I>0.1={100*(v>0.1).mean():.0f}%")


def sec_C(flat, moran, expr, m, ut, hs_default):
    print("\n== C. filter sweep + expression-matched Moran's I gap vs STANCE-only ==")
    stance_only = ut - hs_default
    edges = np.quantile(expr[list(m)], np.linspace(0, 1, 11))

    def matched_gap(S):
        S = [g for g in S if g in m]; gaps = []
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            hsb = [g for g in S if lo <= expr[g] <= hi]
            sob = [g for g in stance_only if lo <= expr[g] <= hi]
            if len(hsb) >= 5 and len(sob) >= 5:
                gaps.append(moran.loc[hsb].median() - moran.loc[sob].median())
        return np.nanmean(gaps) if gaps else np.nan

    print(f"  {'min_pts':>7s} {'min_lfc':>7s} {'nHS':>6s} {'medExpr':>7s} {'medI':>6s} {'matched_gap':>11s}")
    ratio = flat["pts"] / flat["pts_rest"].replace(0, np.nan)
    for mp, mlfc in [(0.8, 1.5), (0.5, 1.0), (0.3, 0.5), (0.1, 0.25)]:
        mask = (flat["pts"] >= mp) & (flat["logfoldchanges"] >= mlfc) & (flat["pvals_adj"] < 0.05)
        mask &= (ratio > 1) | flat["pts_rest"].eq(0)
        S = set(flat.index[mask]) & m
        if not S:
            continue
        print(f"  {mp:>7.2f} {mlfc:>7.2f} {len(S):>6d} {expr.loc[list(S)].median():>7.2f} "
              f"{moran.loc[list(S)].median():>6.3f} {matched_gap(S):>11.3f}")


def sec_D(flat, moran, expr, m, ut, hs_default):
    print("\n== D. recall of STANCE-only as filter drops + HyperSpot DE scores ==")
    stance_only = ut - hs_default
    ratio = flat["pts"] / flat["pts_rest"].replace(0, np.nan); spec = (ratio > 1) | flat["pts_rest"].eq(0)
    for name, mask in [("default(pts.8/lfc1.5/p<.05/spec)",
                        (flat["pts"] >= 0.8) & (flat["logfoldchanges"] >= 1.5) & (flat["pvals_adj"] < 0.05) & spec),
                       ("pts0/lfc0/p<.05/spec", (flat["pvals_adj"] < 0.05) & spec),
                       ("p<.05 only", flat["pvals_adj"] < 0.05),
                       ("no filter (all genes)", pd.Series(True, index=flat.index))]:
        S = set(flat.index[mask]) & m
        r = len(S & stance_only)
        print(f"  {name:34s} nHS={len(S):5d} recall STANCE-only={r}/{len(stance_only)} ({100*r/len(stance_only):.0f}%)")
    print("  DE stats (Wilcoxon z; higher=stronger domain marker):")
    for name, gs in [("HyperSpot-default", [g for g in hs_default if g in flat.index]),
                     ("STANCE-only", [g for g in stance_only if g in flat.index])]:
        f = flat.loc[gs]
        print(f"    {name:18s} med score={f['scores'].median():.2f} med|logfc|={f['logfoldchanges'].abs().median():.2f} "
              f"med pts={f['pts'].median():.2f} %p<.05={100*(f['pvals_adj']<0.05).mean():.0f}%")


def sec_E(flat, moran, expr, m, ut):
    print("\n== E. HyperSpot-only genes at matched significance (p<.05) ==")
    ratio = flat["pts"] / flat["pts_rest"].replace(0, np.nan)
    mask = (flat["pvals_adj"] < 0.05) & ((ratio > 1) | flat["pts_rest"].eq(0))
    hs = set(flat.index[mask]) & m
    hs_only, shared, st_only = hs - ut, hs & ut, ut - hs
    print(f"  shared={len(shared)}  HyperSpot-only={len(hs_only)}  STANCE-only={len(st_only)}")
    for name, gs in [("shared", shared), ("HyperSpot-only", hs_only), ("STANCE-only", st_only)]:
        gs = [g for g in gs if g in m]
        print(f"    {name:16s} n={len(gs):5d} medI={moran.loc[gs].median():.3f} "
              f"medExpr={expr.loc[gs].median():.2f} medScore={flat.loc[gs,'scores'].median():.2f} "
              f"%I>0.1={100*(moran.loc[gs]>0.1).mean():.0f}%")


if __name__ == "__main__":
    aggr, flat, moran, expr, m, ut, hs_default, cnt = load()
    print(f"panel={len(m)}  STANCE utSVGs(in panel)={len(ut)}  HyperSpot default(>=5/10)={len(hs_default)}")
    sec_A(m, ut, cnt)
    sec_B(moran, ut, hs_default, m)
    sec_C(flat, moran, expr, m, ut, hs_default)
    sec_D(flat, moran, expr, m, ut, hs_default)
    sec_E(flat, moran, expr, m, ut)
