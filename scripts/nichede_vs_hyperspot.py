"""
nichede_vs_hyperspot.py — HyperSpot niches vs niche-DE genes (task #13), reproducible.

The comparable quantity: niche-DE gives, for each (index CT A, neighbor CT B), the genes whose A-expression
depends on B being in the neighborhood. HyperSpot gives per-CT spatial DOMAINS grouped into colocalization
NICHES; A's domain that shares a niche with B is A's expression-state "next to B". So HyperSpot's analogue of
niche-DE(A<-B) genes = the marker ct-SVGs of A's domain(s) that colocalize with B in a niche.

For each niche-DE (index, niche) pair with genes, we compute the overlap of niche-DE's gene list with the
markers of the index-CT domains that colocalize with the niche-CT, vs a hypergeometric null over the index-CT
gene panel. Run in the HyperSpot env from demo/:  python ../scripts/nichede_vs_hyperspot.py <Core|Interface>
"""
import sys
import pandas as pd
import numpy as np
from scipy.stats import hypergeom

from hyperspot.metadata import ProjectMetadata
from hyperspot.pipeline.evaluation import ClusterEvaluation
from hyperspot.pipeline.selection import compute_domain_ct_svgs
from hyperspot.utils import load_ct_spatial_data

SECTION = sys.argv[1] if len(sys.argv) > 1 else "Core"
PROJ = f"hyperspot_project_files/Human Kidney Cancer {SECTION}"
NDE = f"/workspace/ctSVG-benchmarking/results/4_Human_Kidney_Cancer_{SECTION}/nicheDE_spotiphy/nicheDE_pair_genes_long.csv"
# niche-DE uses make.names CT labels (T.cell, Epi_non.PT); map back to HyperSpot's
UNDOT = {"B.cell": "B-cell", "T.cell": "T-cell", "Epi_non.PT": "Epi_non-PT"}


def main():
    ProjectMetadata.use(PROJ)
    coloc = ClusterEvaluation().compute_domain_colocalization(unit="spot")
    niche_of = coloc["niche_of"]                       # 'CT|domain' -> niche id

    long = pd.read_csv(NDE)
    long["index"] = long["index"].map(lambda c: UNDOT.get(c, c))
    long["niche"] = long["niche"].map(lambda c: UNDOT.get(c, c))

    # cache index-CT domain markers + gene panel
    marker_cache, panel_cache = {}, {}

    def markers_of(ct, domains):
        if ct not in marker_cache:
            try:
                marker_cache[ct] = compute_domain_ct_svgs(ct, rank_by="scores")
            except Exception:
                marker_cache[ct] = {}
        dm = marker_cache[ct]
        g = set()
        for d in domains:
            key = str(d) if str(d) in {str(k) for k in dm} else d
            for k in dm:
                if str(k) == str(d):
                    g |= set(dm[k]["names"].astype(str))
        return g

    def panel_of(ct):
        if ct not in panel_cache:
            panel_cache[ct] = set(load_ct_spatial_data(ct).var_names.astype(str))
        return panel_cache[ct]

    rows = []
    for (idx, nic), sub in long.groupby(["index", "niche"]):
        # HyperSpot: index-CT domains that share a niche with the neighbour CT
        nic_niches = {niche_of[i] for i in niche_of.index if i.startswith(f"{nic}|")}
        idx_domains = sorted({i.split("|", 1)[1] for i in niche_of.index
                              if i.startswith(f"{idx}|") and niche_of[i] in nic_niches})
        if not idx_domains:
            continue
        panel = panel_of(idx)
        hs = markers_of(idx, idx_domains) & panel
        nde = set(sub["gene"].astype(str)) & panel
        if not hs or not nde:
            continue
        ov = hs & nde
        N = len(panel); exp = len(hs) * len(nde) / N
        p = hypergeom.sf(len(ov) - 1, N, len(nde), len(hs))
        rows.append((idx, nic, len(nde), len(idx_domains), len(hs), len(ov),
                     round(len(ov) / exp, 2) if exp else np.nan, p))

    tab = pd.DataFrame(rows, columns=["index_CT", "neighbor_CT", "nicheDE_genes",
                                      "HS_domains_in_niche", "HS_markers", "overlap",
                                      "fold", "hypergeom_p"]).sort_values("overlap", ascending=False)
    out = f"{PROJ}/nichede_vs_hyperspot_overlap.csv"
    tab.to_csv(out, index=False)
    pd.set_option("display.width", 170)
    print(f"=== {SECTION}: HyperSpot domain-in-niche markers vs niche-DE(index<-neighbor) genes ===")
    print(tab.to_string(index=False))
    sig = tab[tab["hypergeom_p"] < 0.05]
    print(f"\npairs tested: {len(tab)} | significant overlap (p<0.05): {len(sig)} "
          f"| median fold (sig): {sig['fold'].median():.1f}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
