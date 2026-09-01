"""
run_leiden_sweep.py — PCA(20)+Leiden baseline for the domain-subtype comparison.

For each (section, cell type) we cluster the SAME decomposed cells that receive a
transferred scRNA subtype label (via map_domains_to_subtypes), using the standard
non-spatial pipeline: normalize -> log1p -> scale -> PCA(20 comps, to match
node2vec/HyperSpot embedding size) -> kNN -> Leiden. Leiden's `resolution` is
SWEPT so we get a V/ARI-vs-cluster-count curve, and Leiden's `random_state` is
varied 0..9 for a proper seed distribution (unlike MICA, Leiden is seedable).

Also records HyperSpot (10 eval runs) and scMiner (single + any repeats) points so
a standalone preview can overlay all three. Runs in the hyperspot conda env.

Output:
  <proj>/leiden_sweep.csv       long-form: section,ct,resolution,seed,k,homog,compl,v,ari
  <proj>/method_points.csv      HyperSpot/scMiner per-run points for overlay
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.metrics import homogeneity_completeness_v_measure as hcv, adjusted_rand_score as ari

from hyperspot.metadata import ProjectMetadata
from hyperspot.pipeline.selection import map_domains_to_subtypes

SECTIONS = {
    "Interface": ("hyperspot_project_files/Human Kidney Cancer Interface",
                  "spotiphy_project_files/Human Kidney Cancer Interface/Human Kidney Cancer scRNA Ref.h5ad"),
    "Core":      ("hyperspot_project_files/Human Kidney Cancer Core",
                  "spotiphy_project_files/Human Kidney Cancer Core/Human Kidney Cancer scRNA Ref.h5ad"),
}
CTS = ["RCC", "Myeloid", "T-cell"]
# Resolution grid tuned for the DEDUPLICATED graph, where low res -> low k. Spans
# roughly k~2..25 so the curve overlaps HyperSpot (~7-13) and scMiner (~4-8).
RESOLUTIONS = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.6, 2.0, 2.6, 3.4, 4.5, 6.0, 8.0]
RES_FIXED = 1.0   # HyperSpot clusters at Leiden resolution 1.0 -> the apples-to-apples point
N_SEEDS = 10
DEMO = Path("/workspace/HyperSpot/demo")
import scipy.sparse as sp


def dedup_setup(decomp_ct):
    """Deduplicate to unique imputed profiles (same-spot same-CT cells are identical),
    build the standard non-spatial embedding on the UNIQUE profiles, and return the
    prepped AnnData plus `inverse` (per original cell -> unique-profile index) so
    cluster labels can be broadcast back to every cell (as HyperSpot does)."""
    X = decomp_ct.X.toarray() if sp.issparse(decomp_ct.X) else np.asarray(decomp_ct.X)
    Xr = np.round(X, 4)
    uniq, inverse = np.unique(Xr, axis=0, return_inverse=True)   # uniq: (n_unique, genes)
    au = ad.AnnData(uniq.copy())
    if float(X.max()) > 30:
        sc.pp.normalize_total(au, target_sum=1e4)
        sc.pp.log1p(au)
    au.X = np.nan_to_num(au.X)
    sc.pp.scale(au, max_value=10)
    sc.tl.pca(au, n_comps=min(20, au.n_obs - 1, au.n_vars - 1),
              svd_solver="arpack", random_state=0)
    sc.pp.neighbors(au, n_neighbors=15, use_rep="X_pca", random_state=0)
    return au, np.asarray(inverse).ravel()


def main():
    import sys
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    run_sections = {k: v for k, v in SECTIONS.items() if only is None or k in only}
    print("running sections:", list(run_sections), flush=True)
    sweep_rows, point_rows = [], []
    for section, (proj, ref_path) in run_sections.items():
        ProjectMetadata.use(str(DEMO / proj))
        print(f"\n===== {section} =====", flush=True)
        ref = sc.read_h5ad(str(DEMO / ref_path))
        decomp = ad.read_h5ad(str(DEMO / proj / "ST_decomposition.h5ad"))
        for ct in CTS:
            try:
                r = map_domains_to_subtypes(ref, ct, min_cells=80, max_per_subtype=500)
            except Exception as e:
                print(f"  {ct}: skip ({str(e)[:50]})", flush=True); continue
            q = r["query"]
            sub = q.obs["subtype"].astype(str)
            names = pd.Series(q.obs_names.astype(str), index=q.obs_names.astype(str))

            # ---- Leiden+dedup sweep, same labeled cells ----
            # Dedup to unique profiles, cluster those, broadcast labels back to every
            # cell (HyperSpot-style), then score all cells against the transferred subtype.
            dsub = decomp[decomp.obs["cell_type"] == ct].copy()
            dsub.obs_names = dsub.obs_names.astype(str)
            keep = names[names.isin(dsub.obs_names)]
            dsub = dsub[keep.values].copy()
            sub_al = sub.loc[keep.values].values
            au, inverse = dedup_setup(dsub)
            print(f"  {ct}: {dsub.n_obs} cells -> {au.n_obs} unique profiles, "
                  f"sweeping {len(RESOLUTIONS)} res x {N_SEEDS} seeds", flush=True)
            for res in RESOLUTIONS:
                for seed in range(N_SEEDS):
                    sc.tl.leiden(au, resolution=res, random_state=seed,
                                 key_added="lei", flavor="igraph", n_iterations=2, directed=False)
                    lab_u = au.obs["lei"].astype(str).values
                    lab = lab_u[inverse]                       # broadcast to all cells
                    h, c, v = hcv(sub_al, lab)
                    sweep_rows.append((section, ct, res, seed, len(set(lab)), h, c, v, ari(sub_al, lab)))
                    # persist the per-cell Leiden labels at the fixed res (seed 0) for spatial plots
                    if res == RES_FIXED and seed == 0:
                        ldir = Path(DEMO) / proj / "leiden_clusters"; ldir.mkdir(exist_ok=True)
                        pd.DataFrame({"cell_id": keep.values, "leiden": lab}).to_csv(
                            ldir / f"{ct.replace('-', '_')}.csv", index=False)

            # ---- HyperSpot points (10 eval runs) ----
            for i in range(10):
                f = Path(DEMO) / proj / "evaluation" / f"run_{i}" / f"clusters_ct_{ct}.csv"
                if not f.exists():
                    continue
                hg = pd.read_csv(f, index_col=0); hg.index = hg.index.astype(str)
                dom = names.map(hg["domain"].astype(str)); m = dom.notna().values
                h, c, v = hcv(sub[m].values, dom[m].values)
                point_rows.append((section, ct, "HyperSpot", i, int(dom[m].nunique()),
                                   h, c, v, ari(sub[m].values, dom[m].values)))

            # ---- scMiner points (single + repeats) ----
            safe = ct.replace("-", "_")
            sm_single = Path(DEMO) / proj / "scminer_clusters" / f"{safe}.csv"
            sm_files = [("single", sm_single)] + [
                (f"rep{p.parent.name}", p) for p in
                sorted((Path(DEMO) / proj / "scminer_clusters_runs").glob(f"run_*/{safe}.csv"))]
            for tag, p in sm_files:
                if not p.exists():
                    continue
                sm = pd.read_csv(p, index_col=0)["cluster"]
                smc = names.astype("int64").map(sm)
                mm = smc.notna().values
                if mm.sum() == 0:
                    continue
                lab = smc[mm].astype(str).values
                h, c, v = hcv(sub[mm].values, lab)
                point_rows.append((section, ct, "scMiner", tag, len(set(lab)),
                                   h, c, v, ari(sub[mm].values, lab)))

        # write incrementally per section
        sw = pd.DataFrame(sweep_rows, columns=["section", "ct", "resolution", "seed", "k",
                                               "homogeneity", "completeness", "v_measure", "ARI"])
        pt = pd.DataFrame(point_rows, columns=["section", "ct", "method", "run", "k",
                                               "homogeneity", "completeness", "v_measure", "ARI"])
        sw_path = DEMO / "hyperspot_project_files" / "leiden_sweep.csv"
        pt_path = DEMO / "hyperspot_project_files" / "method_points.csv"
        done = set(run_sections)  # replace only the sections computed in this run; keep others
        if sw_path.exists():
            old = pd.read_csv(sw_path)
            sw = pd.concat([old[~old["section"].isin(done)], sw], ignore_index=True)
        if pt_path.exists():
            oldp = pd.read_csv(pt_path)
            pt = pd.concat([oldp[~oldp["section"].isin(done)], pt], ignore_index=True)
        sw.to_csv(sw_path, index=False)
        pt.to_csv(pt_path, index=False)
        print(f"  wrote {len(sw)} sweep rows, {len(pt)} point rows so far", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
