import numpy as np
import pandas as pd
import scanpy as sc

from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score

from ..data import STHypergraphData

from ..metadata import ProjectMetadata

from ..model.hypergraph import HypergraphEmbedderArgs

from ..utils import (
    load_decomposed_st_data,
    load_orig_st_data,
    load_aggr_st_data,
    load_ct_spatial_data,
    var_symbols,
    set_seed,
)

from . import (
    HypergraphEmbedderDriver,
    HyperSpotClustering,
    compute_domain_svgs,
    compute_domain_ct_svgs,
    create_svg_list,
)

def HyperSpot_eval_pipeline(
    run_number: int,
    hypergraph: STHypergraphData,
    args: HypergraphEmbedderArgs,
    num_epochs,
    batch_size,
    accelerator,
    embedding_key="embedding",
    algorithm="leiden",
    min_cluster_pct=-1,
    copy_var_cols=None,
    seed: int = 0,
    n_pca_components: int = 50,
):

    # one seed per pipeline run (train -> embed -> cluster -> score): run_i depends only on its seed
    set_seed(seed)

    project_metadata = ProjectMetadata()

    # representation learning
    embedder = HypergraphEmbedderDriver()
    embedder.fit(
        hypergraph,
        args,
        num_epochs,
        batch_size,
        accelerator,
    )
    embedder.generate_embedding(hypergraph,
                                args,
                                batch_size,
                                accelerator)

    clustering_driver = HyperSpotClustering()

    # clustering tissue domains
    clustering_driver.create_aggr_st_data(var_cols=copy_var_cols)
    clustering_driver.load_embedding_to_aggr_data(embedding_key)
    clustering_driver.compute_spatial_clusters(
        embedding_key=embedding_key,
        algorithm=algorithm,
        min_cluster_pct=min_cluster_pct,
        seed=seed,
        n_pca_components=n_pca_components,
        dedup_spots=True,
    )

    # clustering cell type domains
    clustering_driver.load_embedding_to_ct_spatial_data(embedding_key)
    clustering_driver.compute_ct_spatial_clusters(
        embedding_key=embedding_key,
        algorithm=algorithm,
        min_cluster_pct=min_cluster_pct,
        seed=seed,
        n_pca_components=n_pca_components,
        dedup_spots=True,
    )
    
    run_dir = project_metadata.eval_dir / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # save clusters to eval_dir
    adata_aggr = load_aggr_st_data()
    adata_aggr.obs['domain'].astype(int).to_csv(run_dir / "clusters_ST.csv")

    top_genes = compute_domain_svgs(rank_by='scores')
    top_gene_list = create_svg_list(top_genes)
    top_gene_list.to_csv(run_dir / "svgs_ST.csv")
    
    for ct in project_metadata.ct_list:
        adata_ct = load_ct_spatial_data(ct)
        adata_ct.obs['domain'].astype(int).to_csv(run_dir / f"clusters_ct_{ct}.csv")

        top_ct_genes = compute_domain_ct_svgs(ct, rank_by='scores')
        top_ct_gene_list = create_svg_list(top_ct_genes)
        top_ct_gene_list.to_csv(run_dir / f"svgs_ct_{ct}.csv")

EVAL_SEEDS = [42, 271, 314, 711, 1337, 1618, 1729, 2026, 3407, 9001]

def run_eval(
    hypergraph: STHypergraphData,
    args: HypergraphEmbedderArgs,
    num_epochs,
    batch_size,
    accelerator,
    embedding_key="embedding",
    algorithm="leiden",
    min_cluster_pct=-1,
    n_pca_components=50,
    n=len(EVAL_SEEDS),
):

    n = min(n, len(EVAL_SEEDS))

    # Resume support (gap-filling): recompute exactly the runs in 0..n-1 that are
    # MISSING or INCOMPLETE, not just those past the highest existing index. A run
    # counts as done only if its directory holds the ST outputs plus per-cell-type
    # clusters+svgs for every cell type; a partial directory left by a crashed run
    # (e.g. a sparse-CT failure) is treated as not-done and recomputed. run_i always
    # uses EVAL_SEEDS[i], so seeds stay pinned to their index. Delete any run_<i>
    # directory (or leave a crashed partial one) to force just that run recomputed.
    eval_dir = ProjectMetadata().eval_dir
    ct_list = ProjectMetadata().ct_list

    def _run_complete(i):
        rd = eval_dir / f"run_{i}"
        if not rd.is_dir():
            return False
        needed = ["clusters_ST.csv", "svgs_ST.csv"]
        for ct in ct_list:
            needed += [f"clusters_ct_{ct}.csv", f"svgs_ct_{ct}.csv"]
        return all((rd / f).exists() for f in needed)

    todo = [i for i in range(n) if not _run_complete(i)]

    if not todo:
        print(f"All {n} run(s) present and complete; nothing to do.")
        return

    print(f"Running run(s) {todo} with seeds {[EVAL_SEEDS[i] for i in todo]}.")

    for run_number in todo:
        seed = EVAL_SEEDS[run_number]
        HyperSpot_eval_pipeline(
            run_number,
            hypergraph,
            args,
            num_epochs,
            batch_size,
            accelerator,
            embedding_key=embedding_key,
            algorithm=algorithm,
            min_cluster_pct=min_cluster_pct,
            seed=seed,
            n_pca_components=n_pca_components,
        )

class ClusterEvaluation(object):
    def __init__(self):
        pass

    def _sil(self, X, y_pred):
        return silhouette_score(X, y_pred) if len(np.unique(y_pred)) > 1 else float('nan')

    def _expr_pca(self, adata, n_comps=50):
        """Shared expression PCA space — neutral cross-eval reference for all methods."""
        max_comps = min(adata.n_obs, adata.n_vars) - 1
        n = min(max(1, max_comps), n_comps)
        sc.pp.pca(adata, n_comps=n, mask_var=None)
        return adata.obsm['X_pca']

    # ── Neighborhood-awareness (coupling) ─────────────────────────────────────
    # Representation-level, k-free, no clustering. Replaces silhouette_spatial,
    # which is cluster-count/convexity/expression-autocorrelation confounded.
    # See [[project-hyperspot-neighborhood-eval]].
    def _dcorr(self, X, Y, n_sub=1500, seed=0):
        """Distance correlation between two representations over the same cells.
        Subsamples to n_sub for the O(n^2) distance matrices. Returns dCor in [0,1]."""
        from scipy.spatial.distance import pdist, squareform
        n = X.shape[0]
        if n < 30:
            return float('nan')
        rng = np.random.default_rng(seed)
        if n > n_sub:
            sel = rng.choice(n, n_sub, replace=False)
            X, Y = X[sel], Y[sel]
        def _dcenter(M):
            D = squareform(pdist(M))
            return D - D.mean(0, keepdims=True) - D.mean(1, keepdims=True) + D.mean()
        A, B = _dcenter(X), _dcenter(Y)
        dcov = (A * B).mean()
        va, vb = (A * A).mean(), (B * B).mean()
        if va <= 0 or vb <= 0:
            return float('nan')
        return float(np.sqrt(max(dcov, 0.0) / np.sqrt(va * vb)))

    def _neighbor_enrichment(self, A, ct_of):
        """Cross-type neighbor cell-type enrichment target: log2 P(type|nbhd)/P(type),
        own-CT column zeroed per cell, Laplace-smoothed. A: (n,n) sparse adjacency;
        ct_of: (n,) cell-type per global cell index."""
        import scipy.sparse as sp
        n = A.shape[0]
        types = sorted({t for t in ct_of if t is not None})
        col = {t: j for j, t in enumerate(types)}
        L = sp.lil_matrix((n, len(types)))
        for c in range(n):
            t = ct_of[c]
            if t in col:
                L[c, col[t]] = 1.0
        M = (A @ L.tocsr()).toarray()                    # neighbor CT counts
        label_ct = np.array(types)
        for c in range(n):                               # zero own-CT column
            if ct_of[c] is not None:
                M[c, label_ct == ct_of[c]] = 0.0
        keep = M.sum(0) > 0
        M = M[:, keep]
        N, K = M.shape
        a = 0.5                                          # Laplace pseudocount
        row = M.sum(1, keepdims=True)
        P = (M + a) / (row + a * K)                      # smoothed conditional
        bg = (M.sum(0) + a * N) / (M.sum() + a * N * K)  # smoothed marginal
        return np.log2(P / bg[None, :])

    def compute_coupling(self, neighbor_def='knn', d=20, k=15, min_cells=200, seed=0):
        """Neighborhood-awareness metric (k-free, no clustering): per cell type, does the
        embedding's geometry track spatial-neighborhood cell-type composition BEYOND what
        expression geometry already does?

            coupling_delta = dCor(embedding, nbr) - dCor(expr-PCA(d), nbr)   (>0 = aware)

        neighbor_def='knn' uses each cell's k nearest spatial neighbors (a held-out
        generalization target the model was not trained on). Cell types with < min_cells
        are skipped (overfit). Returns (per_ct dict, mean dict). Replaces silhouette_spatial.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        import scipy.sparse as sp

        pm = ProjectMetadata()
        frames, max_idx = {}, -1
        for ct in pm.ct_list:
            a = load_ct_spatial_data(ct)
            idx = a.obs['cell_idx'].to_numpy().astype(int)
            frames[ct] = (idx, a)
            if len(idx):
                max_idx = max(max_idx, int(idx.max()))
        n_cells = max_idx + 1
        coords = np.zeros((n_cells, 2))
        ct_of = np.array([None] * n_cells, dtype=object)
        for ct, (idx, a) in frames.items():
            coords[idx] = np.asarray(a.obsm['spatial'])
            ct_of[idx] = ct

        if neighbor_def != 'knn':
            raise ValueError("only neighbor_def='knn' is supported here")
        _, ind = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
        rows = np.repeat(np.arange(n_cells), k)
        cols = ind[:, 1:].ravel()
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cells, n_cells))
        Y = self._neighbor_enrichment(A, ct_of)

        results = {}
        for ct in pm.ct_list:
            idx, a = frames[ct]
            if len(idx) < min_cells:
                results[ct] = {'coupling_delta': float('nan'), 'emb_nbr': float('nan'),
                               'exp_nbr': float('nan'), 'emb_exp': float('nan')}
                continue
            E = np.asarray(a.obsm['embedding'])
            Xr = a.X
            Xr = np.asarray(Xr.todense()) if hasattr(Xr, 'todense') else np.asarray(Xr)
            dd = int(min(d, Xr.shape[1], len(idx) - 1))
            Xp = PCA(dd, random_state=seed).fit_transform(StandardScaler().fit_transform(Xr))
            Yct = Y[idx]
            emb_nbr = self._dcorr(E, Yct, seed=seed)
            exp_nbr = self._dcorr(Xp, Yct, seed=seed)
            results[ct] = {'coupling_delta': emb_nbr - exp_nbr, 'emb_nbr': emb_nbr,
                           'exp_nbr': exp_nbr, 'emb_exp': self._dcorr(E, Xp, seed=seed)}
        finite = [v['coupling_delta'] for v in results.values()
                  if np.isfinite(v['coupling_delta'])]
        mean = {'coupling_delta': float(np.mean(finite)) if finite else float('nan'),
                'n_cell_types': len(finite)}
        return results, mean

    def _knn_neighbor_target(self, k=15):
        """Assemble per-CT frames and the global spatial-KNN cross-type neighbor
        enrichment target Y (keyed by cell_idx). Shared by the coupling metrics."""
        from sklearn.neighbors import NearestNeighbors
        import scipy.sparse as sp
        pm = ProjectMetadata()
        frames, max_idx = {}, -1
        for ct in pm.ct_list:
            a = load_ct_spatial_data(ct)
            idx = a.obs['cell_idx'].to_numpy().astype(int)
            frames[ct] = (idx, a)
            if len(idx):
                max_idx = max(max_idx, int(idx.max()))
        n_cells = max_idx + 1
        coords = np.zeros((n_cells, 2))
        ct_of = np.array([None] * n_cells, dtype=object)
        for ct, (idx, a) in frames.items():
            coords[idx] = np.asarray(a.obsm['spatial'])
            ct_of[idx] = ct
        _, ind = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
        A = sp.csr_matrix((np.ones(n_cells * k),
                           (np.repeat(np.arange(n_cells), k), ind[:, 1:].ravel())),
                          shape=(n_cells, n_cells))
        return frames, self._neighbor_enrichment(A, ct_of)

    def compute_coupling_3way(self, scminer_cluster_dir, d=20, k=15, min_cells=200, seed=0):
        """3-way neighborhood-awareness coupling on the same held-out spatial-KNN target:
        HyperSpot embedding vs scMiner node2vec vs expression. Per CT reports each
        representation's dCor to neighborhood and its delta vs expression (hs_delta,
        sm_delta). scMiner node2vec has no spatial input, so sm_delta is expected <= 0.
        Returns (per_ct dict, mean dict)."""
        import re
        from pathlib import Path
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        pm = ProjectMetadata()
        scdir = Path(scminer_cluster_dir)
        frames, Y = self._knn_neighbor_target(k=k)

        results = {}
        for ct in pm.ct_list:
            idx, a = frames[ct]
            if len(idx) < min_cells:
                continue
            Xr = a.X
            Xr = np.asarray(Xr.todense()) if hasattr(Xr, 'todense') else np.asarray(Xr)
            dd = int(min(d, Xr.shape[1], len(idx) - 1))
            Xp = PCA(dd, random_state=seed).fit_transform(StandardScaler().fit_transform(Xr))
            Yct = Y[idx]
            exp_nbr = self._dcorr(Xp, Yct, seed=seed)
            hs_nbr = self._dcorr(np.asarray(a.obsm['embedding']), Yct, seed=seed)
            sm_nbr = float('nan')
            emb_csv = scdir / f"{re.sub(r'[^A-Za-z0-9_]', '_', str(ct))}_embedding.csv"
            if emb_csv.exists():
                sm = pd.read_csv(emb_csv, dtype={'cell_id': str}).set_index('cell_id')
                SM = sm.reindex(a.obs_names.astype(str)).select_dtypes('number').to_numpy()
                if SM.size and not np.isnan(SM).any():
                    sm_nbr = self._dcorr(SM, Yct, seed=seed)
            results[ct] = {'exp_nbr': exp_nbr, 'hs_nbr': hs_nbr, 'sm_nbr': sm_nbr,
                           'hs_delta': hs_nbr - exp_nbr,
                           'sm_delta': (sm_nbr - exp_nbr) if np.isfinite(sm_nbr) else float('nan')}
        hs = [v['hs_delta'] for v in results.values() if np.isfinite(v['hs_delta'])]
        sm = [v['sm_delta'] for v in results.values() if np.isfinite(v['sm_delta'])]
        mean = {'hs_delta': float(np.mean(hs)) if hs else float('nan'),
                'sm_delta': float(np.mean(sm)) if sm else float('nan'),
                'n_cell_types': len(hs)}
        return results, mean

    def compute_reproducibility(self):
        """Reference-free seed reproducibility across the eval runs (evaluation/run_*):
        NO ground truth — measures run-to-run stability of the domains and detected
        (ct)SVGs. For tissue-wide and per cell type:
          cluster_ari     : mean pairwise adjusted Rand index between runs' domain labels
          svg_jaccard     : mean pairwise Jaccard of the detected (ct)SVG gene sets
          svg_containment : mean pairwise |A∩B| / min(|A|,|B|) of those gene sets
        A domain/gene set that reflects real structure (not seed noise) recurs across
        runs -> high values. Returns a dict keyed by 'TISSUE' and each cell type.
        """
        from itertools import combinations
        from sklearn.metrics import adjusted_rand_score
        pm = ProjectMetadata()
        eval_dir = pm.eval_dir
        runs = sorted(int(p.name[4:]) for p in eval_dir.glob("run_*")
                      if p.name[4:].isdigit())

        def _cluster_ari(fname):
            labs = {}
            for r in runs:
                f = eval_dir / f"run_{r}" / fname
                if f.exists():
                    labs[r] = pd.read_csv(f, index_col=0).iloc[:, 0]
            aris = []
            for a, b in combinations(labs, 2):
                common = labs[a].index.intersection(labs[b].index)
                if len(common) >= 2:
                    aris.append(adjusted_rand_score(labs[a].loc[common], labs[b].loc[common]))
            return (float(np.mean(aris)) if aris else float('nan')), len(labs)

        def _geneset_overlap(fname):
            sets = {}
            for r in runs:
                f = eval_dir / f"run_{r}" / fname
                if f.exists():
                    sets[r] = set(pd.read_csv(f, index_col=0).index.astype(str))
            jac, cont = [], []
            for a, b in combinations(sets, 2):
                inter = sets[a] & sets[b]
                union = sets[a] | sets[b]
                m = min(len(sets[a]), len(sets[b]))
                if union:
                    jac.append(len(inter) / len(union))
                if m:
                    cont.append(len(inter) / m)
            return (float(np.mean(jac)) if jac else float('nan'),
                    float(np.mean(cont)) if cont else float('nan'), len(sets))

        results = {}
        a_ari, a_n = _cluster_ari("clusters_ST.csv")
        s_jac, s_cont, s_n = _geneset_overlap("svgs_ST.csv")
        results['TISSUE'] = {'cluster_ari': a_ari, 'svg_jaccard': s_jac,
                             'svg_containment': s_cont, 'n_runs': max(a_n, s_n)}
        for ct in pm.ct_list:
            c_ari, c_n = _cluster_ari(f"clusters_ct_{ct}.csv")
            g_jac, g_cont, g_n = _geneset_overlap(f"svgs_ct_{ct}.csv")
            results[ct] = {'cluster_ari': c_ari, 'svg_jaccard': g_jac,
                           'svg_containment': g_cont, 'n_runs': max(c_n, g_n)}
        return results

    def conserved_core_svgs(self, k=None, save=False):
        """Conserved-core (ct)SVGs: genes detected in >= k of the eval runs — the
        reproducible core (turns compute_reproducibility's containment number into an
        actual gene list). k defaults to a majority of the runs present.

        Returns (cores, summary):
          cores   — {scope: DataFrame[gene, n_runs, frac]} for 'TISSUE' and each cell
                    type, genes with n_runs >= k, sorted by recurrence.
          summary — DataFrame per scope: n_runs present, k used, core_size, and how many
                    genes are 'unanimous' (in every run).
        If save=True, also writes <eval_dir>/conserved_core/<scope>.csv.
        """
        from math import ceil
        from collections import Counter
        import re as _re
        pm = ProjectMetadata()
        eval_dir = pm.eval_dir
        runs = sorted(int(p.name[4:]) for p in eval_dir.glob("run_*")
                      if p.name[4:].isdigit())

        def _core(fname):
            cnt, present = Counter(), 0
            for r in runs:
                f = eval_dir / f"run_{r}" / fname
                if f.exists():
                    present += 1
                    cnt.update(set(pd.read_csv(f, index_col=0).index.astype(str)))
            kk = k if k is not None else ceil(present / 2)
            rows = [(g, c, round(c / present, 3)) for g, c in cnt.items()
                    if present and c >= kk]
            df = (pd.DataFrame(rows, columns=["gene", "n_runs", "frac"])
                  .sort_values(["n_runs", "gene"], ascending=[False, True])
                  .reset_index(drop=True))
            return df, present, kk

        scopes = [("TISSUE", "svgs_ST.csv")] + \
                 [(ct, f"svgs_ct_{ct}.csv") for ct in pm.ct_list]
        out_dir = eval_dir / "conserved_core"
        if save:
            out_dir.mkdir(parents=True, exist_ok=True)
        cores, summary = {}, []
        for scope, fname in scopes:
            df, present, kk = _core(fname)
            cores[scope] = df
            summary.append({"scope": scope, "n_runs": present, "k": kk,
                            "core_size": len(df),
                            "unanimous": int((df["n_runs"] == present).sum()) if present else 0})
            if save and len(df):
                safe = _re.sub(r"[^A-Za-z0-9_]", "_", str(scope))
                df.to_csv(out_dir / f"{safe}.csv", index=False)
        return cores, pd.DataFrame(summary).set_index("scope")

    def compute_domain_colocalization(self, unit="spot", min_support=0.02, lift_thresh=1.5,
                                      cooc_support=0.01, resolution=1.0, seed=0,
                                      max_len=3, save=False, domain_override=None):
        """HyperSpot-native niche recovery: do the *independently-learned* per-CT domains
        colocalize into recurrent multicellular niches?

        ``unit`` sets the transaction (niche) unit:
          * ``'spot'``      — cells sharing a Visium spot (one domain per (spot,CT)).
          * ``'hyperedge'`` — cells in a spot's KNN neighborhood (HyperSpot's *native* unit,
            read from ``hypergraph.pth``); richer, captures the neighborhood composition.

        Pipeline (identical for either unit):
          1. item support; drop items below ``min_support`` (gate rare/unstable domains).
          2. pairwise lift  P(A,B)/(P(A)P(B))  over ``cell_type|domain`` items.
          3. co-occurrence GRAPH (edge iff lift>``lift_thresh`` and joint support>=``cooc_support``,
             weight=log lift) -> Leiden communities = the NICHES.
          4. per-SPOT dominant niche (uniform for either unit — for 'hyperedge' each spot takes
             the mode of the niches of the hyperedges covering it) → spatial niche map.
          5. frequent colocalized (CT,domain) itemsets + association rules (mlxtend fpgrowth).

        This is HyperSpot's OWN approach (graph + itemsets), to be compared against SpaNiche
        (the baseline). It deliberately does NOT use NMF/consensus clustering (SpaNiche's machinery).

        Returns dict: lift, niche_of (Series item->niche), niches (DataFrame), spot_niche
        (DataFrame index spot: niche,location_x,location_y,n_items — always spot-resolved),
        edges (DataFrame source,target,lift,weight — the co-occurrence graph), item_support
        (Series), itemsets, rules, modularity, n_niches, n_spots, n_txn, n_items, unit.
        If save=True, writes <eval_dir>/colocalization/{niches,spot_niche,edges,itemsets,rules}.csv.
        """
        import numpy as np
        from collections import defaultdict
        import igraph as ig
        import leidenalg
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import fpgrowth, association_rules

        pm = ProjectMetadata()
        # ── build transactions: txn_items[i] = item set, txn_spots[i] = spots it covers ──
        spot_xy = {}
        txn_items, txn_spots = [], []
        # domain_override[ct] = domain-label array (positional, per-run domains) for
        # niche reproducibility across eval runs; else use the adata's own 'domain'.
        def _domains(ct, o):
            if domain_override is not None and ct in domain_override:
                return np.asarray(domain_override[ct]).astype(str)
            return o["domain"].astype(str).to_numpy()
        if unit == "spot":
            spot_items = defaultdict(set)
            for ct in map(str, pm.ct_list):
                o = load_ct_spatial_data(ct).obs
                for spot, dom, x, y in zip(o["spot_name"], _domains(ct, o),
                                           o["location_x"], o["location_y"]):
                    spot_items[spot].add(f"{ct}|{dom}")
                    spot_xy.setdefault(spot, (x, y))
            for sp, its in spot_items.items():
                txn_items.append(its); txn_spots.append({sp})
        elif unit == "hyperedge":
            import torch
            hg = torch.load(pm.project_folder / "hypergraph.pth", weights_only=False)
            n_cells = hg["cell_feats"].shape[0]
            item_of = np.empty(n_cells, dtype=object)
            spot_of = hg["cell_to_spot_id"].numpy()
            for ct in map(str, pm.ct_list):
                o = load_ct_spatial_data(ct).obs
                ci = o["cell_idx"].to_numpy()
                item_of[ci] = [f"{ct}|{d}" for d in _domains(ct, o)]
                for sid, x, y in zip(spot_of[ci], o["location_x"], o["location_y"]):
                    spot_xy.setdefault(int(sid), (x, y))
            hi = hg["hyperedge_index"].numpy()
            he_cells = defaultdict(list)
            for c, e in zip(hi[0], hi[1]):
                he_cells[int(e)].append(int(c))
            for e, cells in he_cells.items():
                txn_items.append({item_of[c] for c in cells if item_of[c] is not None})
                txn_spots.append({int(spot_of[c]) for c in cells})
        else:
            raise ValueError("unit must be 'spot' or 'hyperedge'")
        n = len(txn_items)

        # ── item support, gate rare items ───────────────────────────────────────────
        all_items = sorted({i for s in txn_items for i in s})
        supp = {i: sum(i in s for s in txn_items) / n for i in all_items}
        items = [i for i in all_items if supp[i] >= min_support]
        idx = {it: k for k, it in enumerate(items)}
        m = len(items)

        # ── pairwise co-occurrence + lift ───────────────────────────────────────────
        co = np.zeros((m, m))
        for s in txn_items:
            present = [idx[i] for i in s if i in idx]
            for a in range(len(present)):
                for b in range(a + 1, len(present)):
                    co[present[a], present[b]] += 1
                    co[present[b], present[a]] += 1
        p = np.array([supp[i] for i in items])
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = (co / n) / np.outer(p, p)
        lift[~np.isfinite(lift)] = 0.0
        lift_df = pd.DataFrame(lift, index=items, columns=items)

        # ── co-occurrence graph -> Leiden niches ────────────────────────────────────
        gedges, weights = [], []
        for a in range(m):
            for b in range(a + 1, m):
                if lift[a, b] > lift_thresh and co[a, b] >= cooc_support * n:
                    gedges.append((a, b))
                    weights.append(float(np.log(lift[a, b])))
        g = ig.Graph(n=m, edges=gedges)
        g.es["weight"] = weights
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights="weight" if weights else None,
            resolution_parameter=resolution, seed=seed)
        memb = part.membership
        niche_of = pd.Series({items[k]: memb[k] for k in range(m)}, name="niche")
        item_support = pd.Series({items[k]: p[k] for k in range(m)}, name="support")
        edges_df = pd.DataFrame([{"source": items[a], "target": items[b],
                                  "lift": float(lift[a, b]), "weight": w}
                                 for (a, b), w in zip(gedges, weights)])

        niche_rows = []
        for c in sorted(set(memb)):
            mem = [items[k] for k in range(m) if memb[k] == c]
            niche_rows.append({"niche": c, "n_items": len(mem),
                               "n_cell_types": len({it.split("|")[0] for it in mem}),
                               "items": ", ".join(sorted(mem))})
        niches = pd.DataFrame(niche_rows).set_index("niche")

        # ── per-spot dominant niche (uniform: mode of the covering transactions) ─────
        spot_votes = defaultdict(lambda: defaultdict(int))
        spot_union = defaultdict(set)
        for its, sps in zip(txn_items, txn_spots):
            cnt = defaultdict(int)
            for it in its:
                if it in niche_of.index:
                    cnt[niche_of[it]] += 1
            tn = max(cnt, key=cnt.get) if cnt else -1
            for sp in sps:
                if tn != -1:
                    spot_votes[sp][tn] += 1
                spot_union[sp] |= its
        srows = []
        for sp, xy in spot_xy.items():
            votes = spot_votes.get(sp)
            dom_niche = max(votes, key=votes.get) if votes else -1
            srows.append({"spot": sp, "niche": dom_niche, "location_x": xy[0],
                          "location_y": xy[1], "n_items": len(spot_union.get(sp, ()))})
        spot_niche = pd.DataFrame(srows).set_index("spot")

        # ── frequent itemsets + association rules (mlxtend) ─────────────────────────
        te = TransactionEncoder()
        onehot = pd.DataFrame(te.fit(txn_items).transform(txn_items), columns=te.columns_)
        onehot = onehot[[c for c in onehot.columns if c in idx]]  # gated items only
        itemsets = fpgrowth(onehot, min_support=min_support, use_colnames=True, max_len=max_len)
        itemsets = itemsets[itemsets["itemsets"].map(len) >= 2].copy()
        itemsets["k"] = itemsets["itemsets"].map(len)
        itemsets["itemsets"] = itemsets["itemsets"].map(lambda s: ", ".join(sorted(s)))
        itemsets = itemsets.sort_values(["k", "support"], ascending=[True, False]).reset_index(drop=True)
        try:
            rules = association_rules(
                fpgrowth(onehot, min_support=min_support, use_colnames=True, max_len=max_len),
                metric="lift", min_threshold=lift_thresh)
            for col in ("antecedents", "consequents"):
                rules[col] = rules[col].map(lambda s: ", ".join(sorted(s)))
            rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]] \
                .sort_values("lift", ascending=False).reset_index(drop=True)
        except Exception:
            rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

        if save:
            out = pm.eval_dir / "colocalization"
            out.mkdir(parents=True, exist_ok=True)
            tag = "" if unit == "spot" else f"_{unit}"
            niches.to_csv(out / f"niches{tag}.csv")
            spot_niche.to_csv(out / f"spot_niche{tag}.csv")
            edges_df.to_csv(out / f"edges{tag}.csv", index=False)
            itemsets.to_csv(out / f"itemsets{tag}.csv", index=False)
            rules.to_csv(out / f"rules{tag}.csv", index=False)

        return {"lift": lift_df, "niche_of": niche_of, "niches": niches,
                "spot_niche": spot_niche, "edges": edges_df, "item_support": item_support,
                "itemsets": itemsets, "rules": rules,
                "modularity": float(part.modularity), "n_niches": len(set(memb)),
                "n_spots": len(spot_niche), "n_txn": n, "n_items": m, "unit": unit}

    def compute_signaling_niches(self, coloc=None, unit="spot", lift_thresh=1.5, resolution=1.0,
                                 seed=0, min_pts=0.8, min_logfc=1.5, lr_cache=None):
        """ctSVG-GROUNDED signaling niches: a niche = a community of (CT,domain) domains connected by
        BOTH spatial colocalization AND ctSVG-based ligand-receptor signaling.

        Everything is derived from the project's ctSVGs (no raw-expression LR):
          1. per-(CT,domain) ctSVGs (domain markers, ``compute_domain_ct_svgs``) annotated as
             ligand / receptor via the OmniPath LR list.
          2. LR edge A→B if a ligand-ctSVG of A pairs a receptor-ctSVG of B (count of such pairs);
             symmetrized to W_LR[A,B].
          3. two weights per domain-pair: spatial ``lift`` (from ``compute_domain_colocalization``)
             and W_LR — GATED by proximity (edge only where lift>lift_thresh AND W_LR>0).
          4. combined weight = norm(log lift) · norm(W_LR)  (the "strong spatial AND LR" product) →
             Leiden → signaling niches.

        Returns dict: niches (DataFrame), niche_of (Series), spatial_niche_of (spatial-only, for
        comparison), ari_vs_spatial, lr_edges (DataFrame source,target,lr_count,lift,weight),
        lig_of/rec_of ({item: ligand/receptor-ctSVG symbols}), modularity, n_niches, n_edges.
        """
        import numpy as np
        import igraph as ig
        import leidenalg
        from collections import defaultdict
        from sklearn.metrics import adjusted_rand_score
        from .selection import compute_domain_ct_svgs
        pm = ProjectMetadata()

        if coloc is None:
            coloc = self.compute_domain_colocalization(unit=unit, lift_thresh=lift_thresh,
                                                       resolution=resolution, seed=seed)
        items = list(coloc["item_support"].index)
        idx = {it: i for i, it in enumerate(items)}
        m = len(items)
        lift = coloc["lift"].reindex(index=items, columns=items).values

        # ── ctSVGs → ligand/receptor symbol sets, per (CT,domain) ──
        dec = load_decomposed_st_data()
        ensg2sym = dict(zip(dec.var_names.astype(str), dec.var["SYMBOL"].astype(str)))
        lr = self._fetch_omnipath_lr(lr_cache or (pm.eval_dir / "colocalization" / "omnipath_lr.tsv"))
        lig2rec = defaultdict(set)
        for L, R in zip(lr["source_genesymbol"], lr["target_genesymbol"]):
            if isinstance(L, str) and isinstance(R, str):
                lig2rec[L].add(R)
        ligands_all = set(lig2rec)
        receptors_all = set().union(*lig2rec.values()) if lig2rec else set()

        lig_of = {it: set() for it in items}
        rec_of = {it: set() for it in items}
        for ct in sorted({it.split("|")[0] for it in items}):
            for dom, df in compute_domain_ct_svgs(ct, min_pts=min_pts, min_logfc=min_logfc).items():
                it = f"{ct}|{dom}"
                if it not in idx:
                    continue
                genes = df["names"] if "names" in df.columns else df.index
                syms = {ensg2sym.get(str(g), str(g)) for g in genes}
                lig_of[it] = syms & ligands_all
                rec_of[it] = syms & receptors_all

        # ── W_LR[A,B]: # of (ligand-ctSVG of A) → (receptor-ctSVG of B) LR pairs (symmetrized) ──
        Wlr = np.zeros((m, m))
        for a, A in enumerate(items):
            for L in lig_of[A]:
                recs = lig2rec[L]
                for b, B in enumerate(items):
                    if a != b:
                        Wlr[a, b] += len(recs & rec_of[B])
        Wlr = Wlr + Wlr.T

        # ── combine: gate by proximity (lift>thresh) AND signaling (W_LR>0); weight = product ──
        ls = np.log(np.clip(lift, 1e-9, None))
        ls_max = ls[lift > lift_thresh].max() if (lift > lift_thresh).any() else 1.0
        wlr_max = Wlr.max() if Wlr.max() > 0 else 1.0
        edges, weights = [], []
        for a in range(m):
            for b in range(a + 1, m):
                if lift[a, b] > lift_thresh and Wlr[a, b] > 0:
                    edges.append((a, b))
                    weights.append(float((ls[a, b] / ls_max) * (Wlr[a, b] / wlr_max)))
        g = ig.Graph(n=m, edges=edges)
        g.es["weight"] = weights
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                        weights="weight" if weights else None,
                                        resolution_parameter=resolution, seed=seed)
        memb = part.membership
        niche_of = pd.Series({items[k]: memb[k] for k in range(m)}, name="niche")

        spatial_no = coloc["niche_of"]
        common = [it for it in items if it in spatial_no.index]
        ari = adjusted_rand_score([spatial_no[it] for it in common],
                                  [niche_of[it] for it in common]) if common else float("nan")

        niches = pd.DataFrame(
            [{"niche": c, "n_items": sum(x == c for x in memb),
              "n_cell_types": len({items[k].split("|")[0] for k in range(m) if memb[k] == c}),
              "items": ", ".join(sorted(items[k] for k in range(m) if memb[k] == c))}
             for c in sorted(set(memb))]).set_index("niche")
        lr_edges = pd.DataFrame([{"source": items[a], "target": items[b], "lr_count": int(Wlr[a, b]),
                                  "lift": float(lift[a, b]), "weight": w}
                                 for (a, b), w in zip(edges, weights)],
                                columns=["source", "target", "lr_count", "lift", "weight"])
        if len(lr_edges):
            lr_edges = lr_edges.sort_values("weight", ascending=False).reset_index(drop=True)

        # per-spot dominant signaling niche (so plot_niche_domains works on this result)
        spot_votes, spot_xy = defaultdict(lambda: defaultdict(int)), {}
        for ct in sorted({it.split("|")[0] for it in items}):
            o = load_ct_spatial_data(ct).obs
            for spot, dom, x, y in zip(o["spot_name"], o["domain"].astype(str),
                                       o["location_x"], o["location_y"]):
                it = f"{ct}|{dom}"
                if it in niche_of.index:
                    spot_votes[spot][int(niche_of[it])] += 1
                    spot_xy.setdefault(spot, (x, y))
        spot_niche = pd.DataFrame(
            [{"spot": sp, "niche": max(v, key=v.get), "location_x": spot_xy[sp][0],
              "location_y": spot_xy[sp][1], "n_items": sum(v.values())}
             for sp, v in spot_votes.items()]).set_index("spot")

        return {"niches": niches, "niche_of": niche_of, "spatial_niche_of": spatial_no,
                "ari_vs_spatial": float(ari), "lr_edges": lr_edges, "edges": lr_edges,
                "spot_niche": spot_niche, "item_support": coloc["item_support"],
                "lig_of": lig_of, "rec_of": rec_of, "modularity": float(part.modularity),
                "n_niches": len(set(memb)), "n_edges": len(edges), "unit": unit}

    def plot_niche_domains(self, res, niche=None, ncols=5, panel_size=3.0, spot_size=10,
                           cmap="tab10", img="lowres", img_alpha=0.6, show_context=True,
                           save=False, file_ext="png", display=True):
        """(Publication) For each niche, spatial-plot the niche's footprint alongside the
        spatial plot of every member (cell_type, domain), using ``adata_ct``'s own domain
        colors. ``res`` = output of ``compute_domain_colocalization`` (either unit — the
        footprint is drawn from ``res['spot_niche']``, which for unit='hyperedge' already maps
        each niche to the spots its hyperedges cover). ``niche`` limits to one niche (else all).
        ``img`` overlays the H&E tissue behind each panel: 'lowres' (cheap, default), 'hires',
        or None. ``show_context=False`` drops the grey non-member spots/cells, leaving only the
        niche spots (footprint) and only the member-domain cells (member panels) over the tissue.
        Returns list of figures; save=True writes <eval_dir>/colocalization/niche_<u>_<n>.<ext>.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import scanpy as sc
        from math import ceil
        pm = ProjectMetadata()
        sn, niches = res["spot_niche"], res["niches"]
        cmp = plt.get_cmap(cmap)
        targets = [niche] if niche is not None else list(niches.index)

        def _he(ad):
            """(image, extent) for the chosen resolution in full-res spatial coords, or None."""
            if not img:
                return None
            sp = ad.uns.get("spatial", {})
            if not sp:
                return None
            d = next(iter(sp.values()))
            im = d.get("images", {}).get(img)
            sf = d.get("scalefactors", {}).get(f"tissue_{img}_scalef")
            if im is None or sf is None:
                return None
            h, w = im.shape[:2]
            return im, (0, w / sf, h / sf, 0)  # extent with y flipped (origin upper)

        def _bg(ax, he):
            if he is not None:
                ax.imshow(he[0], extent=he[1], alpha=img_alpha, zorder=0)
            else:
                ax.invert_yaxis()  # match Visium orientation when no image is drawn

        # cache per-CT coords + domain colors (scanpy default palette = the "existing" colors)
        cache = {}
        def _ct(ct):
            if ct not in cache:
                ad = load_ct_spatial_data(ct)
                ad.obs["domain"] = ad.obs["domain"].astype("category")
                if "domain_colors" not in ad.uns:
                    sc.pl._utils._set_default_colors_for_categorical_obs(ad, "domain")
                cats = list(ad.obs["domain"].cat.categories)
                cache[ct] = (ad.obsm["spatial"], ad.obs["domain"].to_numpy(),
                             dict(zip(cats, ad.uns["domain_colors"])), _he(ad))
            return cache[ct]

        figs = []
        for nb in targets:
            members = niches.loc[nb, "items"].split(", ")
            n_panel = 1 + len(members)
            ncol = min(ncols, n_panel)
            nrow = ceil(n_panel / ncol)
            fig, axes = plt.subplots(nrow, ncol, figsize=(panel_size * ncol, panel_size * nrow),
                                     squeeze=False)
            axes = axes.ravel()
            # panel 0 — niche spatial footprint (spots whose dominant niche == nb)
            ax0 = axes[0]
            _bg(ax0, _ct(members[0].split("|", 1)[0])[3])  # tissue image (shared across CTs)
            if show_context:
                ax0.scatter(sn["location_x"], sn["location_y"], s=spot_size, c="lightgrey",
                            linewidths=0, zorder=1)
            inn = sn[sn["niche"] == nb]
            ax0.scatter(inn["location_x"], inn["location_y"], s=spot_size,
                        color=cmp(nb % 10), linewidths=0, zorder=2)
            ax0.set_title(f"niche {nb}  ({len(inn)} spots)", fontsize=9, fontweight="bold")
            ax0.set_aspect("equal"); ax0.axis("off")
            # member panels — each (ct, domain) in its adata_ct color, others grey
            for j, mem in enumerate(members, start=1):
                ct, dom = mem.split("|", 1)
                XY, doms, colors, he = _ct(ct)
                ax = axes[j]
                _bg(ax, he)
                mask = doms == dom
                if show_context:
                    ax.scatter(XY[~mask, 0], XY[~mask, 1], s=spot_size * 0.6, c="lightgrey",
                               linewidths=0, zorder=1)
                ax.scatter(XY[mask, 0], XY[mask, 1], s=spot_size * 0.6,
                           color=colors.get(dom, "red"), linewidths=0, zorder=2)
                ax.set_title(mem, fontsize=9)
                ax.set_aspect("equal"); ax.axis("off")
            for j in range(n_panel, len(axes)):
                fig.delaxes(axes[j])
            fig.suptitle(f"Niche {nb} — {niches.loc[nb, 'n_cell_types']} cell types, "
                         f"{len(members)} (CT,domain) members", fontsize=11, fontweight="bold")
            fig.set_layout_engine("tight")
            if save:
                out = pm.eval_dir / "colocalization"
                out.mkdir(parents=True, exist_ok=True)
                fig.savefig(out / f"niche_{res['unit']}_{nb}.{file_ext}",
                            dpi=200, bbox_inches="tight")
            figs.append(fig)
            if not display:
                plt.close(fig)
        return figs

    def plot_niche_graph(self, res, figsize=(11, 11), node_scale=6000, edge_scale=1.2,
                         font_size=7, cmap="tab10", seed=0, title=None, hide_isolated=False,
                         save=False, file_ext="png", display=True):
        """(Publication) The Leiden graph WITH edges: nodes = (CT,domain) items colored by niche
        and sized by support; edge width ∝ weight. Works on ``compute_domain_colocalization`` output
        (co-occurrence graph) or ``compute_signaling_niches`` output (ctSVG-LR signaling graph — pass
        a ``title``). Returns the figure.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import networkx as nx
        pm = ProjectMetadata()
        edges, niche_of, supp = res["edges"], res["niche_of"], res["item_support"]
        cmp = plt.get_cmap(cmap)

        G = nx.Graph()
        G.add_nodes_from(niche_of.index)
        for _, r in edges.iterrows():
            G.add_edge(r["source"], r["target"], weight=r["weight"])
        if hide_isolated:                              # drop unconnected domains → show the connections
            G.remove_nodes_from([n for n in list(G.nodes()) if G.degree(n) == 0])
        pos = nx.spring_layout(G, weight="weight", seed=seed, k=1.6 / max(len(G), 1) ** 0.5)

        nodes = list(G.nodes())
        node_colors = [cmp(int(niche_of[i]) % 10) for i in nodes]
        node_sizes = [node_scale * float(supp.get(i, 0.01)) + 40 for i in nodes]

        fig, ax = plt.subplots(figsize=figsize)
        elist = list(G.edges())
        if elist:                                      # rescale widths to a visible range; color by niche
            w = np.array([G[u][v]["weight"] for u, v in elist])
            wn = (w - w.min()) / (w.max() - w.min() + 1e-9)
            widths = 0.5 + edge_scale * 4.0 * wn
            ecolors = [cmp(int(niche_of[u]) % 10) if niche_of[u] == niche_of[v]
                       else (0.6, 0.6, 0.6, 0.35) for u, v in elist]
            nx.draw_networkx_edges(G, pos, edgelist=elist, width=widths, edge_color=ecolors,
                                   alpha=0.7, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=node_colors,
                               node_size=node_sizes, linewidths=0.5, edgecolors="white", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)
        present = sorted({int(niche_of[i]) for i in nodes})
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmp(c % 10),
                          markersize=9, label=f"niche {c}") for c in present]
        ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)
        ax.set_title(title or (f"Cross-CT domain co-occurrence graph ({res['unit']} unit) — "
                     f"{res['n_niches']} niches (Leiden, modularity={res['modularity']:.2f})"),
                     fontsize=11)
        ax.axis("off")
        fig.set_layout_engine("tight")
        if save:
            out = pm.eval_dir / "colocalization"
            out.mkdir(parents=True, exist_ok=True)
            tag = "signaling" if title else res["unit"]
            fig.savefig(out / f"niche_graph_{tag}.{file_ext}", dpi=200, bbox_inches="tight")
        if not display:
            plt.close(fig)
        return fig

    def sweep_colocalization_resolution(self, unit="hyperedge", resolutions=None, n_seeds=8,
                                        min_support=0.02, lift_thresh=1.5, cooc_support=0.01):
        """Sweep Leiden resolution on the colocalization graph to find a STABLE niche count.
        Builds the co-occurrence graph once (via ``compute_domain_colocalization``), then
        re-partitions at each resolution × ``n_seeds``. Pick the resolution at the modularity
        plateau with high ``seed_ARI`` (partitions reproducible across seeds).

        Returns DataFrame [resolution, n_niches, n_niches_multicellular, modularity, seed_ARI].
        """
        import numpy as np
        import igraph as ig
        import leidenalg
        from itertools import combinations
        from sklearn.metrics import adjusted_rand_score
        if resolutions is None:
            resolutions = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
        res = self.compute_domain_colocalization(unit=unit, min_support=min_support,
                                                 lift_thresh=lift_thresh, cooc_support=cooc_support)
        items = list(res["item_support"].index)
        idx = {it: i for i, it in enumerate(items)}
        ed = res["edges"]
        g = ig.Graph(n=len(items))
        wkey = None
        if len(ed):
            g.add_edges([(idx[s], idx[t]) for s, t in zip(ed["source"], ed["target"])])
            g.es["weight"] = list(ed["weight"]); wkey = "weight"
        rows = []
        for r in resolutions:
            membs = [np.array(leidenalg.find_partition(
                        g, leidenalg.RBConfigurationVertexPartition, weights=wkey,
                        resolution_parameter=r, seed=s).membership) for s in range(n_seeds)]
            ncl = [len(set(mb)) for mb in membs]
            ncl_mc = [sum(1 for c in set(mb) if list(mb).count(c) > 1) for mb in membs]
            mods = [g.modularity(mb, weights=wkey) for mb in membs]
            aris = [adjusted_rand_score(membs[i], membs[j])
                    for i, j in combinations(range(n_seeds), 2)]
            rows.append({"resolution": r, "n_niches": int(np.median(ncl)),
                         "n_niches_multicellular": int(np.median(ncl_mc)),
                         "modularity": round(float(np.mean(mods)), 3),
                         "seed_ARI": round(float(np.mean(aris)), 3)})
        return pd.DataFrame(rows)

    # ── niche validation (reference-free) ────────────────────────────────────────
    def niche_reproducibility(self, unit="spot", **coloc_kwargs):
        """(Validation) Niche STABILITY across the stochastic pipeline: rebuild the niches
        from each eval run's per-CT domains (``run_*/clusters_ct_*.csv``) and compare the
        spot→niche labelings pairwise with ARI. High = niches are a stable property of the
        tissue, not an artifact of one random seed. Deeper than the Leiden-seed sweep (which
        fixes the graph); here the embedding→domains→graph→niches all vary per run.

        Returns dict: mean_ari, std, min, max, n_runs, n_common_spots.
        """
        import numpy as np
        from itertools import combinations
        from sklearn.metrics import adjusted_rand_score
        pm = ProjectMetadata()
        runs = sorted(int(p.name[4:]) for p in pm.eval_dir.glob("run_*") if p.name[4:].isdigit())
        cts = list(map(str, pm.ct_list))
        run_sn = []
        for r in runs:
            dom = {ct: pd.read_csv(pm.eval_dir / f"run_{r}" / f"clusters_ct_{ct}.csv",
                                   index_col=0)["domain"].to_numpy()
                   for ct in cts if (pm.eval_dir / f"run_{r}" / f"clusters_ct_{ct}.csv").exists()}
            res = self.compute_domain_colocalization(unit=unit, domain_override=dom, **coloc_kwargs)
            run_sn.append(res["spot_niche"]["niche"].to_dict())
        common = sorted(set.intersection(*[set(d) for d in run_sn]))
        mats = [np.array([d[s] for s in common]) for d in run_sn]
        aris = [adjusted_rand_score(mats[i], mats[j]) for i, j in combinations(range(len(runs)), 2)]
        return {"mean_ari": float(np.mean(aris)), "std": float(np.std(aris)),
                "min": float(np.min(aris)), "max": float(np.max(aris)),
                "n_runs": len(runs), "n_common_spots": len(common)}

    def niche_spatial_coherence(self, res, k=6, n_perm=500, seed=0):
        """(Validation) Are niche spots spatially CONTIGUOUS? Fraction of each spot's k
        spatial-KNN neighbors that share its niche, vs a label-shuffled null → z-score.
        High z = niches are coherent territories, not scattered. ``res`` = compute output.

        Returns dict: observed, null_mean, null_std, z, k, n_spots.
        """
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        sn = res["spot_niche"]
        coords = sn[["location_x", "location_y"]].to_numpy()
        lab = sn["niche"].to_numpy()
        nbr = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)[1][:, 1:]
        same = lambda l: float(np.mean(l[nbr] == l[:, None]))
        obs = same(lab)
        rng = np.random.default_rng(seed)
        null = np.array([same(rng.permutation(lab)) for _ in range(n_perm)])
        return {"observed": obs, "null_mean": float(null.mean()), "null_std": float(null.std()),
                "z": float((obs - null.mean()) / null.std()), "k": k, "n_spots": len(lab)}

    def _fetch_omnipath_lr(self, cache_path):
        """Ligand-receptor interactions from the OmniPath REST service (the python client is
        broken on numpy 2 via ``np.NAN``, so query the HTTP API directly). Cached to disk."""
        import urllib.request
        from pathlib import Path
        cache_path = Path(cache_path)
        if not cache_path.exists():
            url = ("https://omnipathdb.org/interactions?genesymbols=1&datasets="
                   "ligrecextra,cellphonedb,cellchatdb,celltalkdb,connectomedb,icellnet,"
                   "guide2pharma&fields=curation_effort&format=tsv")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=90) as resp:
                cache_path.write_bytes(resp.read())
        return pd.read_csv(cache_path, sep="\t")

    def niche_ligrec(self, res, n_perms=1000, threshold=0.1, corr_method="fdr_bh", seed=0,
                     lr_cache=None):
        """(Validation, task #11) Do the niches capture ligand-receptor (cell-cell-communication)
        structure, learned with NO LR prior? Runs ``squidpy.gr.ligrec`` (permutation null + FDR)
        with ``cluster_key='niche'`` on spot expression, fed the OmniPath LR list. Reports the
        WITHIN- vs BETWEEN-niche significant-interaction rate — within > between means niches are
        coherent signaling units. ``res`` = compute output (its ``spot_niche`` labels the spots).

        Returns dict: within_rate, between_rate, enrichment, n_lr_tested, per_niche_sig (Series),
        hub_niche, top_pairs_hub (Series), pvalues (DataFrame).
        """
        import numpy as np
        import squidpy as sq
        pm = ProjectMetadata()
        aggr = load_aggr_st_data()
        dec = load_decomposed_st_data()
        ensg2sym = dict(zip(dec.var_names.astype(str), dec.var["SYMBOL"].astype(str)))
        syms = pd.Index([ensg2sym.get(g, g) for g in aggr.var_names.astype(str)])
        first = ~syms.duplicated()
        aggr = aggr[:, first].copy(); aggr.var_names = syms[first]
        s2n = res["spot_niche"]["niche"].to_dict()
        if res.get("unit") == "hyperedge":
            # hyperedge spot_niche is keyed by integer spot-id → map back to Visium barcodes
            import torch
            spot_of = torch.load(pm.project_folder / "hypergraph.pth",
                                 weights_only=False)["cell_to_spot_id"].numpy()
            id2bc = {}
            for ct in map(str, pm.ct_list):
                o = load_ct_spatial_data(ct).obs
                for ci, bc in zip(o["cell_idx"].to_numpy(), o["spot_name"].astype(str)):
                    id2bc.setdefault(int(spot_of[ci]), bc)
            s2n = {id2bc[k]: v for k, v in s2n.items() if k in id2bc}
        s2n = {str(k): v for k, v in s2n.items()}
        aggr.obs["niche"] = pd.Categorical(
            [f"N{s2n.get(str(s), -1)}" for s in aggr.obs_names.astype(str)])
        aggr = aggr[aggr.obs["niche"] != "N-1"].copy()
        if aggr.n_obs == 0:
            raise ValueError(
                "niche_ligrec: no spots matched res['spot_niche'] — its index doesn't align to "
                "the Visium barcodes. Pass a FRESH result from compute_domain_colocalization; if "
                "you re-ran notebook cells your `_coloc` may be stale (Restart & Run All).")

        cache = lr_cache or (pm.eval_dir / "colocalization" / "omnipath_lr.tsv")
        lr = self._fetch_omnipath_lr(cache)
        inter = (lr[["source_genesymbol", "target_genesymbol"]].dropna().drop_duplicates()
                 .rename(columns={"source_genesymbol": "source", "target_genesymbol": "target"}))
        sq.gr.ligrec(aggr, cluster_key="niche", interactions=inter, use_raw=False,
                     n_perms=n_perms, seed=seed, threshold=threshold, corr_method=corr_method,
                     copy=False, key_added="nlr")
        pv = aggr.uns["nlr"]["pvalues"]
        sig = (pv < 0.05).values
        cols = list(pv.columns)
        within = np.array([a == b for a, b in cols])
        nlr = pv.shape[0]
        wr = sig[:, within].sum() / (nlr * within.sum())
        br = sig[:, ~within].sum() / (nlr * (~within).sum())
        per_niche = pd.Series({cols[i][0]: int(sig[:, i].sum())
                               for i in np.where(within)[0]}).sort_values(ascending=False)
        tn = per_niche.idxmax()
        col = [i for i in np.where(within)[0] if cols[i][0] == tn][0]
        top = pv.iloc[:, col].sort_values().head(10)
        return {"within_rate": float(wr), "between_rate": float(br),
                "enrichment": float(wr / br), "n_lr_tested": int(nlr),
                "per_niche_sig": per_niche, "hub_niche": tn, "top_pairs_hub": top, "pvalues": pv}

    def niche_lr_sources(self, res, ligrec_res, pval_thresh=0.05, save=False):
        """Trace each niche's significant WITHIN-niche LR pairs back to the sender/receiver
        (cell_type, domain) members: **sender** = the niche member with the highest LIGAND
        expression, **receiver** = the member with the highest RECEPTOR expression. This is an
        expression attribution layered on the spot-level ``niche_ligrec`` test (interpretation
        that resolves directionality — NOT an independently-tested member-level p-value).

        ``res`` = ``compute_domain_colocalization`` output; ``ligrec_res`` = ``niche_ligrec`` output.
        Returns DataFrame [niche, ligand, receptor, sender, receiver, L_expr, R_expr, pvalue],
        sorted by niche then pvalue. save=True → evaluation/colocalization/niche_lr_sources.csv.
        """
        import numpy as np
        import scipy.sparse as sp
        pm = ProjectMetadata()
        niches, pv = res["niches"], ligrec_res["pvalues"]
        within = {a: i for i, (a, b) in enumerate(pv.columns) if a == b}

        # significant within-niche LR pairs per niche + the genes they need
        niche_pairs, needed = {}, set()
        for n in niches.index:
            col = within.get(f"N{n}")
            if col is None:
                continue
            series = pv.iloc[:, col]
            pairs = [((s, t), float(p)) for (s, t), p in series.items()
                     if isinstance(s, str) and isinstance(t, str) and s and t
                     and pd.notna(p) and p < pval_thresh]
            niche_pairs[n] = pairs
            for (s, t), _ in pairs:
                needed.update([s, t])

        # per-(CT,domain) mean expression for the needed genes (from cell-level per-CT adata)
        members = set()
        for n in niche_pairs:
            members.update(niches.loc[n, "items"].split(", "))
        ct_doms = {}
        for m in members:
            ct, dom = m.split("|", 1)
            ct_doms.setdefault(ct, set()).add(dom)
        memexpr = {}
        for ct, doms in ct_doms.items():
            ad = load_ct_spatial_data(ct)
            symcol = {}
            for i, s in enumerate(ad.var["SYMBOL"].astype(str)):
                if s in needed:
                    symcol.setdefault(s, i)
            dvals = ad.obs["domain"].astype(str).to_numpy()
            for dom in doms:
                mask = dvals == dom
                if not symcol or mask.sum() == 0:
                    memexpr[(ct, dom)] = {}
                    continue
                sub = ad.X[mask]
                sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
                mu = np.asarray(sub).mean(0).ravel()
                memexpr[(ct, dom)] = {g: float(mu[c]) for g, c in symcol.items()}

        def top_member(mems, gene):
            best, bv = None, -np.inf
            for m in mems:
                ct, dom = m.split("|", 1)
                v = memexpr.get((ct, dom), {}).get(gene, np.nan)
                if not np.isnan(v) and v > bv:
                    best, bv = m, v
            return best, (round(bv, 3) if best is not None else np.nan)

        rows = []
        for n, pairs in niche_pairs.items():
            mems = niches.loc[n, "items"].split(", ")
            for (L, R), p in pairs:
                sm, sv = top_member(mems, L)
                rm, rv = top_member(mems, R)
                rows.append({"niche": n, "ligand": L, "receptor": R, "sender": sm,
                             "receiver": rm, "L_expr": sv, "R_expr": rv, "pvalue": p})
        df = (pd.DataFrame(rows)
              .sort_values(["niche", "pvalue", "L_expr"], ascending=[True, True, False])
              .reset_index(drop=True))
        if save:
            out = pm.eval_dir / "colocalization"
            out.mkdir(parents=True, exist_ok=True)
            df.to_csv(out / "niche_lr_sources.csv", index=False)
        return df

    def _niche_pair_sources(self, res, triples):
        """For (niche, ligand, receptor) triples return [(sender, receiver)] = the niche member
        (cell_type|domain) with the highest ligand / receptor mean expression (from per-CT adata)."""
        import numpy as np
        import scipy.sparse as sp
        from collections import defaultdict
        niches = res["niches"]
        need, nmembers = defaultdict(set), {}
        for nb, L, R in triples:
            mems = niches.loc[nb, "items"].split(", ")
            nmembers[nb] = mems
            for m in mems:
                need[m.split("|", 1)[0]].update([L, R])
        memexpr = {}
        for ct, genes in need.items():
            ad = load_ct_spatial_data(ct)
            symcol = {}
            for i, s in enumerate(ad.var["SYMBOL"].astype(str)):
                if s in genes:
                    symcol.setdefault(s, i)
            dvals = ad.obs["domain"].astype(str).to_numpy()
            for dom in set(dvals):
                mask = dvals == dom
                if not symcol or mask.sum() == 0:
                    continue
                sub = ad.X[mask]
                sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
                mu = np.asarray(sub).mean(0).ravel()
                memexpr[(ct, dom)] = {g: float(mu[c]) for g, c in symcol.items()}

        def top(mems, gene):
            best, bv = None, -np.inf
            for m in mems:
                ct, dom = m.split("|", 1)
                v = memexpr.get((ct, dom), {}).get(gene, -np.inf)
                if v > bv:
                    best, bv = m, v
            return best
        return [(top(nmembers[nb], L), top(nmembers[nb], R)) for nb, L, R in triples]

    def plot_niche_signaling(self, src, by="CT", ncols=4, panel_size=3.4, cmap="magma",
                             save=False, file_ext="png", display=True):
        """(Publication) Per-niche directed signaling heatmap: sender (rows) × receiver (cols),
        colored by the number of significant LR pairs (from ``niche_lr_sources``). ``by='CT'``
        (cell type, compact) or ``'member'`` (cell_type|domain, detailed). Shows the full
        sender→receiver structure — incl. cell types absent from the top-ranked pairs. Returns fig.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from math import ceil
        pm = ProjectMetadata()
        d = src.copy()
        if by == "CT":
            d["S"] = d["sender"].str.split("|").str[0]
            d["R"] = d["receiver"].str.split("|").str[0]
        else:
            d["S"], d["R"] = d["sender"], d["receiver"]
        cats = sorted(set(d["S"].dropna()) | set(d["R"].dropna()))
        niches = sorted(d["niche"].unique())
        ncol = min(ncols, len(niches))
        nrow = ceil(len(niches) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(panel_size * ncol, panel_size * nrow),
                                 squeeze=False, layout="constrained")
        axes = axes.ravel()
        vmax = 0
        mats = {}
        for n in niches:
            sub = d[d["niche"] == n]
            M = (pd.crosstab(sub["S"], sub["R"]).reindex(index=cats, columns=cats, fill_value=0))
            mats[n] = M; vmax = max(vmax, M.values.max())
        for i, n in enumerate(niches):
            ax = axes[i]
            im = ax.imshow(mats[n].values, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=90, fontsize=6)
            ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=6)
            ax.set_title(f"N{n}", fontsize=10, fontweight="bold")
            ax.set_xlabel("receiver", fontsize=7); ax.set_ylabel("sender", fontsize=7)
        for j in range(len(niches), len(axes)):
            fig.delaxes(axes[j])
        fig.colorbar(im, ax=axes[:len(niches)].tolist(), shrink=0.6, label="# significant LR pairs")
        fig.suptitle(f"Niche directed signaling ({by}-level): sender → receiver",
                     fontsize=12, fontweight="bold")
        if save:
            out = pm.eval_dir / "colocalization"
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"niche_signaling_{by}.{file_ext}", dpi=200, bbox_inches="tight")
        if not display:
            plt.close(fig)
        return fig

    def plot_lr_spatial(self, pairs, img="lowres", img_alpha=0.6, spot_size=16,
                        panel_size=3.6, cmap="magma", show_context=True, save=False,
                        file_ext="png", display=True):
        """(Publication) Side-by-side SPATIAL map for selected niche LR pairs: the **sender**
        member's cells colored by **ligand** expression | the **receiver** member's cells colored
        by **receptor** expression (member-restricted; only the sender/receiver (CT,domain) cells
        are colored, tissue H&E behind). Lets you literally see the ligand-high sender region sit
        next to the receptor-high receiver region — the spatial basis of the interaction.

        ``pairs`` = a SUBSET of ``niche_lr_sources`` rows (cols niche, ligand, receptor, sender,
        receiver) — e.g. ``_src.groupby('niche').head(1)`` for the top pair per niche. One row of
        two panels per pair. Returns the figure.
        """
        import numpy as np
        import scipy.sparse as sp
        import matplotlib.pyplot as plt
        pm = ProjectMetadata()
        pairs = pairs.reset_index(drop=True)
        n = len(pairs)
        fig, axes = plt.subplots(n, 2, figsize=(panel_size * 2, panel_size * n),
                                 squeeze=False, layout="constrained")

        cache = {}
        def _ct(ct):
            if ct not in cache:
                ad = load_ct_spatial_data(ct)
                symcol = {}
                for i, s in enumerate(ad.var["SYMBOL"].astype(str)):
                    symcol.setdefault(s, i)
                he = None
                if img:
                    spd = ad.uns.get("spatial", {})
                    if spd:
                        d = next(iter(spd.values()))
                        im = d.get("images", {}).get(img)
                        sf = d.get("scalefactors", {}).get(f"tissue_{img}_scalef")
                        if im is not None and sf is not None:
                            h, w = im.shape[:2]
                            he = (im, (0, w / sf, h / sf, 0))
                cache[ct] = (ad.obsm["spatial"], ad.obs["domain"].astype(str).to_numpy(),
                             ad.X, symcol, he)
            return cache[ct]

        def _panel(ax, member, gene, title):
            ct, dom = member.split("|", 1)
            XY, doms, X, symcol, he = _ct(ct)
            if he is not None:
                ax.imshow(he[0], extent=he[1], alpha=img_alpha, zorder=0)
            else:
                ax.invert_yaxis()
            mask = doms == dom
            if show_context:
                ax.scatter(XY[~mask, 0], XY[~mask, 1], s=spot_size * 0.4, c="lightgrey",
                           linewidths=0, zorder=1)
            col = symcol.get(gene)
            if col is not None and mask.sum() > 0:
                sub = X[mask][:, col]
                vals = (sub.toarray().ravel() if sp.issparse(sub) else np.asarray(sub).ravel())
                pts = ax.scatter(XY[mask, 0], XY[mask, 1], s=spot_size, c=vals, cmap=cmap,
                                 linewidths=0, zorder=2)
                fig.colorbar(pts, ax=ax, shrink=0.7)
            else:
                ax.scatter(XY[mask, 0], XY[mask, 1], s=spot_size, c="grey", linewidths=0, zorder=2)
            ax.set_title(title, fontsize=9)
            ax.set_aspect("equal"); ax.axis("off")

        for i, row in pairs.iterrows():
            _panel(axes[i, 0], row["sender"], row["ligand"],
                   f"N{row['niche']}  ligand {row['ligand']}\nsender {row['sender']}")
            _panel(axes[i, 1], row["receiver"], row["receptor"],
                   f"N{row['niche']}  receptor {row['receptor']}\nreceiver {row['receiver']}")
        fig.suptitle("LR spatial — ligand in sender  |  receptor in receiver",
                     fontsize=12, fontweight="bold")
        if save:
            out = pm.eval_dir / "colocalization"
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"lr_spatial.{file_ext}", dpi=200, bbox_inches="tight")
        if not display:
            plt.close(fig)
        return fig

    def hyperedge_lr(self, res=None, per_niche=True, min_frac=0.1, corr_method="fdr_bh",
                     lr_cache=None, save=False):
        """Neighborhood-level LR test (spatially strict): do ligand and receptor co-occur in the
        SAME hyperedge (a spot + its KNN neighborhood = paracrine range) more than chance? ENFORCES
        the proximity that ``niche_ligrec`` only assumes, using HyperSpot's hypergraph as the unit.

        **Default ``per_niche=True``**: the test is run WITHIN EACH NICHE'S hyperedges (a hyperedge
        belongs to the majority niche of its cells' spots), so it is proximity-strict AND niche-linked
        — "niche N's neighborhoods concentrate LR pair X". Needs a niche result ``res`` (from
        ``compute_domain_colocalization``; auto-computed spot-unit if None). ``per_niche=False`` gives
        the tissue-wide version (niche-agnostic — a statement about the embedding's neighborhoods).

        Per gene a hyperedge is 'active' if ≥ ``min_frac`` of its cells express it (>0).
          * per_niche=True — **DIFFERENTIAL / niche-specific**: ``lift = (both-active rate in niche N) /
            (both-active rate tissue-wide)``; hypergeometric p that N's hyperedges are ENRICHED for the
            pair vs the whole tissue (FDR per niche). A pair ranks high in N only if it co-occurs MORE in
            N than elsewhere — so lift>1 means niche-concentrated (fixes the within-niche-baseline
            disconnect where a pair rare-but-co-occurring in N yet abundant OUTSIDE scored high).
          * per_niche=False — tissue-wide ``lift = P(L&R same hyperedge)/(P(L)P(R))`` vs independence.

        per_niche=True  → dict: table [niche, ligand, receptor, n_both (in N), n_both_tissue, lift,
                          pvalue, padj], summary (per-niche n_hyperedges/n_pairs/frac_sig/median_lift),
                          frac_sig, median_lift, n_pairs, n_hyperedges.
        per_niche=False → dict: table [ligand, receptor, n_both, n_L, n_R, lift, pvalue, padj], frac_sig,
                          median_lift, n_pairs, n_hyperedges.
        NOTE hyperedges overlap (share cells) → the hypergeometric p is approximate; lift is the robust
        effect size (shuffle null = rigorous follow-up).
        """
        import numpy as np
        import scipy.sparse as sp
        import torch
        from collections import defaultdict
        from scipy.stats import hypergeom
        from statsmodels.stats.multitest import multipletests
        pm = ProjectMetadata()

        hg = torch.load(pm.project_folder / "hypergraph.pth", weights_only=False)
        hi = hg["hyperedge_index"].numpy()
        n_cells = hg["cell_feats"].shape[0]
        n_HE = int(hi[1].max()) + 1
        spot_of = hg["cell_to_spot_id"].numpy()
        inc = sp.csr_matrix((np.ones(hi.shape[1], np.float32), (hi[0], hi[1])),
                            shape=(n_cells, n_HE))            # cells × hyperedges

        dec = load_decomposed_st_data()
        order = np.argsort(dec.obs_names.astype(int).to_numpy())   # align rows to cell_idx order
        X = dec.X[order]
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        binexpr = (X > 0).astype(np.float32)
        counts = (inc.T @ binexpr)
        counts = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
        sizes = np.asarray(inc.T.sum(1)).ravel()
        active = (counts / np.maximum(sizes[:, None], 1)) >= min_frac   # bool HE × genes

        sym = dec.var["SYMBOL"].astype(str).to_numpy()
        sym2col = {}
        for i, s in enumerate(sym):
            sym2col.setdefault(s, i)
        cache = lr_cache or (pm.eval_dir / "colocalization" / "omnipath_lr.tsv")
        lr = self._fetch_omnipath_lr(cache)
        lp = lr[["source_genesymbol", "target_genesymbol"]].dropna().drop_duplicates()
        pairs = [(str(L), str(R), sym2col[str(L)], sym2col[str(R)])
                 for L, R in zip(lp["source_genesymbol"], lp["target_genesymbol"])
                 if str(L) in sym2col and str(R) in sym2col and sym2col[str(L)] != sym2col[str(R)]]

        def _test(act):
            m = act.shape[0]
            rows = []
            for L, R, lc, rc in pairs:
                aL, aR = act[:, lc], act[:, rc]
                nL, nR = int(aL.sum()), int(aR.sum())
                if nL == 0 or nR == 0:
                    continue
                O = int((aL & aR).sum())
                E = nL * nR / m
                rows.append((L, R, O, nL, nR, (O / E) if E > 0 else np.nan,
                             float(hypergeom.sf(O - 1, m, nL, nR))))
            df = pd.DataFrame(rows, columns=["ligand", "receptor", "n_both", "n_L", "n_R",
                                             "lift", "pvalue"])
            if len(df):
                df["padj"] = multipletests(df["pvalue"].values, method=corr_method)[1]
            return df

        out = pm.eval_dir / "colocalization"
        if not per_niche:
            df = _test(active).sort_values("lift", ascending=False).reset_index(drop=True)
            if save:
                out.mkdir(parents=True, exist_ok=True)
                df.to_csv(out / "hyperedge_lr.csv", index=False)
            return {"table": df, "frac_sig": float(((df.padj < 0.05) & (df.lift > 1)).mean()),
                    "median_lift": float(df.lift.median()), "n_pairs": len(df), "n_hyperedges": n_HE}

        # ── per-niche: label each hyperedge by the majority niche of its cells' spots ──
        if res is None:
            res = self.compute_domain_colocalization(unit="spot")
        s2n = res["spot_niche"]["niche"].to_dict()
        if res.get("unit") == "spot":                              # spot_id → niche via barcode
            sid2n = {}
            for ct in map(str, pm.ct_list):
                o = load_ct_spatial_data(ct).obs
                for ci, bc in zip(o["cell_idx"].to_numpy(), o["spot_name"].astype(str)):
                    if bc in s2n:
                        sid2n.setdefault(int(spot_of[ci]), s2n[bc])
        else:
            sid2n = {int(k): v for k, v in s2n.items()}
        cell_niche = np.array([sid2n.get(int(spot_of[c]), -1) for c in range(n_cells)])
        he_cells = defaultdict(list)
        for c, e in zip(hi[0], hi[1]):
            he_cells[int(e)].append(int(c))
        he_niche = np.full(n_HE, -1)
        for e, cells in he_cells.items():
            v = cell_niche[cells]; v = v[v >= 0]
            if len(v):
                he_niche[e] = np.bincount(v).argmax()

        # DIFFERENTIAL per-niche test: is the pair's co-occurrence ENRICHED in niche N's hyperedges
        # vs the WHOLE tissue (hypergeometric)? lift = (both-rate in N)/(both-rate tissue-wide).
        # This is niche-SPECIFIC (marker-style) — a pair only ranks high in N if it co-occurs MORE in
        # N than elsewhere, resolving the "within-niche-baseline" disconnect (a pair rare-but-co-
        # occurring in N, yet abundant OUTSIDE, no longer scores high in N).
        lcols = np.array([p[2] for p in pairs]); rcols = np.array([p[3] for p in pairs])
        lig = np.array([p[0] for p in pairs]); rec = np.array([p[1] for p in pairs])
        both_mat = active[:, lcols] & active[:, rcols]          # n_HE × n_pairs
        both_total = both_mat.sum(0).astype(int)                # per pair, tissue-wide
        tables, summ = [], []
        for nb in sorted(set(int(x) for x in he_niche if x >= 0)):
            mask = he_niche == nb; n_in = int(mask.sum())
            both_in = both_mat[mask].sum(0).astype(int)
            exp = both_total * (n_in / n_HE)
            with np.errstate(divide="ignore", invalid="ignore"):
                lift = np.where(exp > 0, both_in / exp, np.nan)
            pval = hypergeom.sf(both_in - 1, n_HE, both_total, n_in)   # P(>= both_in) vs tissue
            keep = both_total > 0
            df = pd.DataFrame({"niche": nb, "ligand": lig, "receptor": rec, "n_both": both_in,
                               "n_both_tissue": both_total, "lift": lift, "pvalue": pval})[keep]
            if not len(df):
                continue
            df["padj"] = multipletests(df["pvalue"].values, method=corr_method)[1]
            df = df.sort_values("lift", ascending=False)
            tables.append(df)
            summ.append({"niche": nb, "n_hyperedges": n_in, "n_pairs": len(df),
                         "frac_sig": round(float(((df.padj < 0.05) & (df.lift > 1)).mean()), 3),
                         "median_lift": round(float(df.lift.median()), 3)})
        table = pd.concat(tables).reset_index(drop=True)
        summary = pd.DataFrame(summ).set_index("niche")
        if save:
            out.mkdir(parents=True, exist_ok=True)
            table.to_csv(out / "hyperedge_lr_per_niche.csv", index=False)
        return {"table": table, "summary": summary, "n_hyperedges": n_HE, "n_pairs": len(table),
                "frac_sig": float(((table.padj < 0.05) & (table.lift > 1)).mean()),
                "median_lift": float(table.lift.median())}

    def plot_lr_cooccurrence(self, pairs=None, hlr=None, res=None, top_per_niche=1,
                             restrict_to_niche=True, show_niche_outline=True, dim_alpha=0.2,
                             niche_style="mask", mode="bivariate", annotate_ct=True,
                             level="neighborhood", k=6,
                             threshold=0.0, img="lowres", img_alpha=0.55, spot_size=6,
                             panel_size=4.0, ncols=3, cmap="magma", save=False, file_ext="png",
                             display=True):
        """(Publication) Co-occurrence map per LR pair. Three ``mode``s:
          * ``'bivariate'`` (default) — CONTINUOUS 2-D color of (ligand, receptor) intensity: blue =
            ligand, red = receptor, orange = both, light = neither; shade AND size ∝ level (a 2-D color
            key is drawn). Not binary — a weakly-expressing spot is pale, a strong one is saturated.
          * ``'intensity'`` — single HEATMAP of ``√(neighborhood ligand · receptor)`` (high only where
            BOTH are strong), per-panel colorbar.
          * ``'category'`` — discrete LIGAND only / RECEPTOR only / BOTH / neither (binary threshold).
        ``level``: 'neighborhood' (default — present if IT or any of ``k`` spatial neighbors express
        it; paracrine range) or 'spot'. ``annotate_ct`` puts the sender/receiver **cell type** (top
        ligand / receptor expresser among niche members) in each title.

        **Niche-aware (default when ``hlr`` given):** pass ``hlr`` = a PER-NICHE ``hyperedge_lr`` result
        and ``res`` = ``compute_domain_colocalization`` output → plots each niche's top ``top_per_niche``
        pair(s); ``restrict_to_niche=True`` colors only that niche's spots. When False (tissue-wide),
        ``niche_style`` marks the niche: ``'mask'`` (default — a transparent, dark-bordered segmentation
        contour around the niche), ``'dim'`` (dim spots outside via ``dim_alpha``), or ``'none'``. Or pass
        explicit ``pairs`` for tissue-wide maps. Returns the figure.
        """
        import numpy as np
        import scipy.sparse as sp
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from math import ceil
        pm = ProjectMetadata()

        # ── assemble panels: each = (niche | None, ligand, receptor, lift | None) ──
        panels = []
        if hlr is not None:
            t = hlr["table"]
            sig = t[(t["padj"] < 0.05) & (t["lift"] > 1)]
            src = sig if len(sig) else t
            for nb, g in src.groupby("niche"):
                for _, row in g.sort_values("lift", ascending=False).head(top_per_niche).iterrows():
                    panels.append((int(nb), row["ligand"], row["receptor"], float(row["lift"])))
        elif pairs is not None:
            if hasattr(pairs, "columns"):
                pairs = list(zip(pairs["ligand"], pairs["receptor"]))
            panels = [(None, L, R, None) for (L, R) in (tuple(p) for p in pairs)]
        else:
            raise ValueError("plot_lr_cooccurrence: pass `pairs` or a per-niche `hlr` result.")

        aggr = load_aggr_st_data()
        X = aggr.X.toarray() if sp.issparse(aggr.X) else np.asarray(aggr.X)
        XY = (aggr.obsm["spatial"] if "spatial" in aggr.obsm
              else aggr.obs[["location_x", "location_y"]].to_numpy())
        nbr = None
        if level == "neighborhood":
            from sklearn.neighbors import NearestNeighbors
            nbr = NearestNeighbors(n_neighbors=k + 1).fit(XY).kneighbors(XY)[1]  # incl self
        dec = load_decomposed_st_data()
        ensg2sym = dict(zip(dec.var_names.astype(str), dec.var["SYMBOL"].astype(str)))
        sym = np.array([ensg2sym.get(g, g) for g in aggr.var_names.astype(str)])
        sym2col = {}
        for i, s in enumerate(sym):
            sym2col.setdefault(s, i)

        # per-spot niche — map res.spot_niche onto aggr barcodes (for restrict and/or outline)
        niche_of_spot = None
        has_niche = any(p[0] is not None for p in panels)
        if has_niche and res is not None:
            s2n = res["spot_niche"]["niche"].to_dict()
            if res.get("unit") == "hyperedge":
                import torch
                spot_of = torch.load(pm.project_folder / "hypergraph.pth",
                                     weights_only=False)["cell_to_spot_id"].numpy()
                id2bc = {}
                for ct in map(str, pm.ct_list):
                    o = load_ct_spatial_data(ct).obs
                    for ci, bc in zip(o["cell_idx"].to_numpy(), o["spot_name"].astype(str)):
                        id2bc.setdefault(int(spot_of[ci]), bc)
                s2n = {id2bc[kk]: v for kk, v in s2n.items() if kk in id2bc}
            s2n = {str(kk): v for kk, v in s2n.items()}
            niche_of_spot = np.array([s2n.get(b, -1) for b in aggr.obs_names.astype(str)])

        # sender/receiver cell type per panel (top ligand / receptor expresser in the niche)
        sr = {}
        if annotate_ct and res is not None and has_niche:
            keyed = [(nb, L, R) for (nb, L, R, _) in panels if nb is not None]
            for (nb, L, R), (sm, rm) in zip(keyed, self._niche_pair_sources(res, keyed)):
                sr[(nb, L, R)] = (sm, rm)

        he = None
        if img:
            spd = aggr.uns.get("spatial", {})
            if spd:
                d = next(iter(spd.values()))
                im = d.get("images", {}).get(img)
                sf = d.get("scalefactors", {}).get(f"tissue_{img}_scalef")
                if im is not None and sf is not None:
                    h, w = im.shape[:2]
                    he = (im, (0, w / sf, h / sf, 0))

        palette = {-1: "#f3f4f6", 0: "#d1d5db", 1: "#2563eb", 2: "#dc2626", 3: "#f59e0b"}
        sizemap = {-1: spot_size * 0.25, 0: spot_size * 0.35, 1: spot_size, 2: spot_size, 3: spot_size * 1.7}
        # bivariate color: continuous 2-D map of (ligand, receptor) intensity
        import matplotlib.colors as mcolors
        _corner = {k: np.array(mcolors.to_rgb(h)) for k, h in
                   {"00": "#e5e7eb", "10": "#2563eb", "01": "#dc2626", "11": "#f59e0b"}.items()}

        def _norm(v):                                  # robust 0..1 (98th pct of nonzero)
            pos = v[v > 0]
            hi = np.percentile(pos, 98) if pos.size else 1.0
            return np.clip(v / (hi + 1e-12), 0.0, 1.0)

        def _bivar(ln, rn):                            # bilinear blend of the 4 corner colors
            ln = np.asarray(ln)[:, None]; rn = np.asarray(rn)[:, None]
            return ((1 - ln) * (1 - rn) * _corner["00"] + ln * (1 - rn) * _corner["10"]
                    + (1 - ln) * rn * _corner["01"] + ln * rn * _corner["11"])

        n = len(panels); ncol = min(ncols, n); nrow = ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(panel_size * ncol, panel_size * nrow),
                                 squeeze=False, layout="constrained")
        axes = axes.ravel()
        def _ctname(m):
            return m.split("|", 1)[0] if m else None

        def _niche_mask(ax, innic):                 # transparent, bordered segmentation of the niche
            from scipy.ndimage import gaussian_filter
            xe = np.linspace(XY[:, 0].min(), XY[:, 0].max(), 120)
            ye = np.linspace(XY[:, 1].min(), XY[:, 1].max(), 120)
            Hn = np.histogram2d(XY[innic, 0], XY[innic, 1], bins=[xe, ye])[0]
            Ha = np.histogram2d(XY[:, 0], XY[:, 1], bins=[xe, ye])[0]
            frac = gaussian_filter(Hn, 1.2) / (gaussian_filter(Ha, 1.2) + 1e-9)
            xc = (xe[:-1] + xe[1:]) / 2; yc = (ye[:-1] + ye[1:]) / 2
            ax.contourf(xc, yc, frac.T, levels=[0.5, 1.01], colors=["#1e293b"], alpha=0.08, zorder=1.5)
            ax.contour(xc, yc, frac.T, levels=[0.5], colors=["#0f172a"], linewidths=1.3,
                       alpha=0.85, zorder=6)

        for i, (nb, L, R, lift) in enumerate(panels):
            ax = axes[i]
            if he is not None:
                ax.imshow(he[0], extent=he[1], alpha=img_alpha, zorder=0)
            else:
                ax.invert_yaxis()
            lc, rc = sym2col.get(L), sym2col.get(R)
            sm, rm = sr.get((nb, L, R), (None, None))
            if nb is not None:
                head = f"N{nb}: {L}" + (f" ({_ctname(sm)})" if sm else "") + \
                       f" → {R}" + (f" ({_ctname(rm)})" if rm else "") + \
                       (f"   lift {lift:.1f}" if lift else "")
            else:
                head = f"{L} → {R}"
            if lc is None or rc is None:
                ax.set_title(head + "\n(not in panel)", fontsize=8); ax.axis("off"); continue
            innic = (niche_of_spot == nb) if (nb is not None and niche_of_spot is not None) else None
            # tissue-wide view: mark the niche (bordered mask overlay, or dim outside)
            alpha = None
            if innic is not None and not restrict_to_niche and show_niche_outline and niche_style == "dim":
                alpha = np.where(innic, 1.0, dim_alpha)

            if mode == "bivariate":
                Ln, Rn = X[:, lc], X[:, rc]
                if nbr is not None:
                    Ln, Rn = X[:, lc][nbr].mean(1), X[:, rc][nbr].mean(1)
                ln, rn = _norm(Ln), _norm(Rn)
                rgb = _bivar(ln, rn)
                sizes = spot_size * (0.3 + 1.0 * np.maximum(ln, rn))   # size ∝ intensity too
                show = innic if (restrict_to_niche and innic is not None) else np.ones(len(ln), bool)
                rgba = rgb[show]
                if alpha is not None:
                    rgba = np.concatenate([rgba, alpha[show, None]], axis=1)
                ax.scatter(XY[show, 0], XY[show, 1], c=rgba, s=sizes[show], linewidths=0, zorder=2)
                ax.set_title(head, fontsize=8)
            elif mode == "intensity":
                Ln, Rn = X[:, lc], X[:, rc]
                if nbr is not None:
                    Ln, Rn = X[:, lc][nbr].mean(1), X[:, rc][nbr].mean(1)
                score = np.sqrt(np.clip(Ln, 0, None) * np.clip(Rn, 0, None))
                pos = score > 0
                if restrict_to_niche and innic is not None:
                    pos = pos & innic
                ax.scatter(XY[~pos, 0], XY[~pos, 1], s=spot_size * 0.3, c="#e5e7eb",
                           linewidths=0, zorder=1)
                if pos.any():
                    a = alpha[pos] if alpha is not None else None
                    pts = ax.scatter(XY[pos, 0], XY[pos, 1], s=spot_size, c=score[pos], cmap=cmap,
                                     vmin=0, alpha=a, linewidths=0, zorder=3)
                    fig.colorbar(pts, ax=ax, shrink=0.6)
                ax.set_title(head, fontsize=8)
            else:
                aL, aR = X[:, lc] > threshold, X[:, rc] > threshold
                if nbr is not None:
                    aL, aR = aL[nbr].any(1), aR[nbr].any(1)
                cat = np.where(aL & aR, 3, np.where(aL, 1, np.where(aR, 2, 0)))
                if restrict_to_niche and innic is not None:
                    cat = np.where(innic, cat, -1)
                for c in (-1, 0, 1, 2, 3):     # 'both' drawn last, on top
                    m = cat == c
                    if m.any():
                        a = alpha[m] if alpha is not None else None
                        ax.scatter(XY[m, 0], XY[m, 1], s=sizemap[c], c=palette[c],
                                   alpha=a, linewidths=0, zorder=2 + c)
                ax.set_title(head + f"   (both: {int((cat == 3).sum())})", fontsize=8)
            if innic is not None and not restrict_to_niche and show_niche_outline and niche_style == "mask":
                _niche_mask(ax, innic)
            ax.set_aspect("equal"); ax.axis("off")
        empty = list(range(n, len(axes)))
        if mode == "bivariate" and empty:            # 2-D color key in the first spare cell
            lax = axes[empty.pop(0)]
            gsz = 32
            gy, gx = np.mgrid[0:gsz, 0:gsz] / (gsz - 1)
            grid = _bivar(gx.ravel(), gy.ravel()).reshape(gsz, gsz, 3)
            lax.imshow(grid, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
            lax.set_xlabel("ligand →", fontsize=9); lax.set_ylabel("receptor →", fontsize=9)
            lax.set_xticks([]); lax.set_yticks([]); lax.set_title("color key", fontsize=9)
        for j in empty:
            fig.delaxes(axes[j])
        handles = []
        if mode == "category":
            labels = {1: "ligand only", 2: "receptor only", 3: "both (signaling)"}
            if has_niche and restrict_to_niche:
                labels = {**labels, -1: "outside niche"}
            handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[kk],
                              markersize=9, label=v) for kk, v in labels.items()]
        if has_niche and not restrict_to_niche and show_niche_outline:
            if niche_style == "mask":
                handles.append(Line2D([0], [0], color="#0f172a", lw=1.5, label="niche boundary"))
            elif niche_style == "dim":
                handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#6b7280",
                                      alpha=dim_alpha + 0.2, markersize=9, label="dimmed = outside niche"))
        if handles:
            fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                       frameon=False, fontsize=9)
        _cbar = {"intensity": "√(ligand·receptor) intensity",
                 "bivariate": "ligand (blue) · receptor (red) · both (orange), shade ∝ intensity"
                 }.get(mode, "ligand / receptor / both")
        fig.suptitle(f"LR neighborhood co-occurrence — {_cbar}"
                     + (" (per niche)" if has_niche else ""),
                     fontsize=12, fontweight="bold")
        if save:
            out = pm.eval_dir / "colocalization"
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"lr_cooccurrence{'_per_niche' if has_niche else ''}.{file_ext}",
                        dpi=200, bbox_inches="tight")
        if not display:
            plt.close(fig)
        return fig

    def _compute_metrics_baseline(self, adata):
        """PCA + Leiden: home eval = expression PCA space (same as cross-eval by definition)."""
        X_pca = self._expr_pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, flavor='igraph', n_iterations=2)
        y_pred = adata.obs['leiden'].to_numpy()
        S = adata.obsm.get('spatial')
        return {
            'silhouette_pca':     self._sil(X_pca, y_pred),
            'silhouette_spatial': self._sil(S, y_pred) if S is not None else float('nan'),
        }

    def _compute_metrics_embedding(self, adata, clusters):
        """HyperSpot: home eval in embedding space; cross-eval: same expression PCA as baseline."""
        y_pred = clusters['domain'].to_numpy()
        E = adata.obsm.get('embedding')
        S = adata.obsm.get('spatial')
        if E is None:
            raise ValueError("adata.obsm['embedding'] not found — run load_embedding_to_ct_spatial_data first.")
        X_pca = self._expr_pca(adata)
        return {
            'silhouette_emb':     self._sil(E, y_pred),
            'silhouette_expr':    self._sil(X_pca, y_pred),   # cross-eval: HyperSpot clusters in expression PCA
            'silhouette_spatial': self._sil(S, y_pred) if S is not None else float('nan'),
        }

    def _compute_mean_ct_scores(self, ct_results, key):
        items = ct_results.values() if isinstance(ct_results, dict) else ct_results
        vals = [r[key] for r in items if r is not None and key in r]
        finite = [v for v in vals if not np.isnan(v)]
        return np.mean(finite) if finite else float('nan')
    
    def _compute_run_scores_summary(self, run_results, key):
        mean_ct_scores = [self._compute_mean_ct_scores(run_result, key) for run_result in run_results]
        
        return {
            'mean': np.mean(mean_ct_scores),
            'std': np.std(mean_ct_scores)
        }
    
    def compute_ST_baseline_scores(self):
        project_metadata = ProjectMetadata()

        adata_st = load_orig_st_data()
        adata_aggr = load_aggr_st_data()

        common_spots = adata_st.obs_names.intersection(adata_aggr.obs_names)
        adata_st = adata_st[common_spots].copy()
        adata_st.X = adata_st.X.todense()
        
        return self._compute_metrics_baseline(adata_st)

    def compute_CT_baseline_scores(self, n_jobs=30):
        project_metadata = ProjectMetadata()

        adata_decomp = load_decomposed_st_data()
        ct_key = project_metadata.ct_key
        
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_metrics_baseline)(
                adata_decomp[adata_decomp.obs[ct_key] == ct].copy()
            ) for ct in project_metadata.ct_list
        )
        ct_baseline_results = {
            ct: r for ct, r in zip(project_metadata.ct_list, results_list)
        }
        mean_ct_baseline_results = {
            metric: self._compute_mean_ct_scores(ct_baseline_results, metric)
            for metric in next(iter(ct_baseline_results.values())).keys()
        }
        return ct_baseline_results, mean_ct_baseline_results
        
    def compute_ST_embedding_scores(self, num_runs=10, n_jobs=30):
        project_metadata = ProjectMetadata()
        
        adata_st = load_orig_st_data()
        adata_aggr = load_aggr_st_data()

        common_spots = adata_st.obs_names.intersection(adata_aggr.obs_names)
        adata_st = adata_st[common_spots].copy()
        adata_st.X = adata_st.X.todense()
        
        run_dirs = [project_metadata.eval_dir / f"run_{run_id}" for run_id in range(num_runs)]
        st_run_clusters = (pd.read_csv(run_dir / "clusters_ST.csv") for run_dir in run_dirs) 
        run_results = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_metrics_embedding)(
                adata_aggr,
                st_run_cluster
            ) for st_run_cluster in st_run_clusters)

        metrics = run_results[0].keys()
        run_scores_summary = {
            metric: {
                'mean': np.mean([r[metric] for r in run_results if metric in r]),
                'std':  np.std([r[metric] for r in run_results if metric in r])
            }
            for metric in metrics
        }
        return run_scores_summary, run_results

    def compute_CT_spotiphy_scores(self, scminer_cluster_dir, n_jobs=30):
        """
        Evaluate Spotiphy iscRNA + MICA cluster quality per cell type.

        Reads per-CT cluster CSVs produced by scripts/run_scminer_clustering.py
        (run in scMiner env) and computes silhouette, DBCVI, RSI in the same
        spaces as compute_CT_baseline_scores and compute_CT_embedding_scores,
        enabling a direct three-way comparison.

        Parameters
        ----------
        scminer_cluster_dir : str or Path
            Directory containing <ct_safe>.csv files from run_scminer_clustering.py.
        n_jobs : int
            Parallel workers.

        Returns
        -------
        ct_results : dict[ct → dict[metric → float]]
        mean_results : dict[metric → float]
        """
        from pathlib import Path

        project_metadata = ProjectMetadata()
        cluster_dir = Path(scminer_cluster_dir)
        adata_decomp = load_decomposed_st_data()
        ct_key = project_metadata.ct_key

        def _eval_one_ct(ct):
            import re
            safe = re.sub(r"[^A-Za-z0-9_]", "_", ct)
            csv_path  = cluster_dir / f"{safe}.csv"
            emb_path  = cluster_dir / f"{safe}_embedding.csv"
            if not csv_path.exists():
                print(f"  [{ct}] no MICA cluster file found — skipping", flush=True)
                return None
            clusters = pd.read_csv(csv_path, dtype={"cell_id": str}).set_index("cell_id")
            adata_ct = adata_decomp[adata_decomp.obs[ct_key] == ct].copy()
            common   = adata_ct.obs_names.intersection(clusters.index)
            adata_ct = adata_ct[common].copy()
            clusters = clusters.loc[common]
            # load MICA embedding coords if saved
            if emb_path.exists():
                emb_df = pd.read_csv(emb_path, dtype={"cell_id": str}).set_index("cell_id")
                emb_df = emb_df.loc[common]
                # use 20D node2vec if available (dim_0..dim_N), else fall back to 2D XY
                dim_cols = [c for c in emb_df.columns if c.startswith("dim_")]
                coord_cols = dim_cols if dim_cols else ["X", "Y"]
                adata_ct.obsm['embedding'] = emb_df[coord_cols].values
            S = adata_ct.obsm.get('spatial')
            E = adata_ct.obsm.get('embedding')
            y_pred = clusters['cluster'].to_numpy()
            return {
                'silhouette_mica':    self._sil(E, y_pred) if E is not None else float('nan'),
                'silhouette_spatial': self._sil(S, y_pred) if S is not None else float('nan'),
            }

        ct_list = project_metadata.ct_list
        ct_results_list = Parallel(n_jobs=n_jobs)(
            delayed(_eval_one_ct)(ct) for ct in ct_list
        )
        ct_results = {
            ct: res for ct, res in zip(ct_list, ct_results_list) if res is not None
        }
        mean_results = {
            metric: self._compute_mean_ct_scores(ct_results, metric)
            for metric in next(iter(ct_results.values())).keys()
        }
        return ct_results, mean_results

    def compute_CT_embedding_scores(self, num_runs=10, n_jobs=-1):
        project_metadata = ProjectMetadata()
        ct_list  = project_metadata.ct_list
        run_dirs = [project_metadata.eval_dir / f"run_{run_id}" for run_id in range(num_runs)]

        all_ct_run_results = []
        for run_dir in run_dirs:
            adata_cts      = (load_ct_spatial_data(ct).copy() for ct in ct_list)
            ct_run_clusters = (pd.read_csv(run_dir / f"clusters_ct_{ct}.csv") for ct in ct_list)
            results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(self._compute_metrics_embedding)(adata_ct, clusters)
                for adata_ct, clusters in zip(adata_cts, ct_run_clusters)
            )
            all_ct_run_results.append({ct: r for ct, r in zip(ct_list, results)})

        sample_metrics = next(iter(next(iter(all_ct_run_results)).values())).keys()
        run_scores_summary = {
            metric: self._compute_run_scores_summary(all_ct_run_results, metric)
            for metric in sample_metrics
        }

        return all_ct_run_results, run_scores_summary
