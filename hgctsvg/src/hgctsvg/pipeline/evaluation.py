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
)

from . import (
    HypergraphEmbedderDriver,
    HGCTSVGClustering,
    compute_domain_svgs,
    compute_domain_ct_svgs,
    create_svg_list,
)

def HGCTSVG_eval_pipeline(
    run_number: int,
    args: HypergraphEmbedderArgs,
    num_epochs,
    batch_size,
    accelerator,
    embedding_key="embedding",
    algorithm="leiden",
    min_cluster_pct=0.05,
    copy_var_cols=None,
    seed: int = 0,
    n_pca_components: int = 50,
):

    project_metadata = ProjectMetadata()

    # representation learning
    hypergraph = STHypergraphData.load()
    embedder = HypergraphEmbedderDriver()
    embedder.fit(
        hypergraph,
        args,
        num_epochs,
        batch_size,
        accelerator,
        seed=seed,
    )
    embedder.generate_embedding(hypergraph,
                                args,
                                batch_size,
                                accelerator)

    clustering_driver = HGCTSVGClustering()

    # clustering tissue domains
    clustering_driver.create_aggr_st_data(var_cols=copy_var_cols)
    clustering_driver.load_embedding_to_aggr_data(embedding_key)
    clustering_driver.compute_spatial_clusters(
        embedding_key=embedding_key,
        algorithm=algorithm,
        min_cluster_pct=-1,
        seed=seed,
        n_pca_components=n_pca_components,
    )

    # clustering cell type domains
    clustering_driver.load_embedding_to_ct_spatial_data(embedding_key)
    clustering_driver.compute_ct_spatial_clusters(
        embedding_key=embedding_key,
        algorithm=algorithm,
        min_cluster_pct=min_cluster_pct,
        seed=seed,
        n_pca_components=n_pca_components,
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
    args: HypergraphEmbedderArgs,
    num_epochs,
    batch_size,
    accelerator,
    embedding_key="embedding",
    algorithm="leiden",
    min_cluster_pct=0.05,
    n_pca_components=50,
    n=len(EVAL_SEEDS),
):

    n = min(n, len(EVAL_SEEDS))

    # Resume support: a run_<i> directory is only created once that run's
    # clustering completes, so existing directories mark finished runs. Continue
    # from the next index after the highest existing one, reusing the seed tied
    # to that index (run_i always uses EVAL_SEEDS[i]). Delete a run_<i> directory
    # to force that run to be recomputed on the next invocation.
    eval_dir = ProjectMetadata().eval_dir
    existing = [
        int(p.name[len("run_"):])
        for p in eval_dir.glob("run_*")
        if p.is_dir() and p.name[len("run_"):].isdigit()
    ]
    start = max(existing) + 1 if existing else 0

    if start >= n:
        print(f"Found {len(existing)} existing run(s) (up to run_{max(existing)}); "
              f"nothing to do for num_runs={n}.")
        return

    print(f"Resuming model evaluation: running run_{start}..run_{n - 1} "
          f"with seeds {EVAL_SEEDS[start:n]}.")

    for run_number in range(start, n):
        seed = EVAL_SEEDS[run_number]
        HGCTSVG_eval_pipeline(
            run_number,
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
        """HGCTSVG: home eval in embedding space; cross-eval: same expression PCA as baseline."""
        y_pred = clusters['domain'].to_numpy()
        E = adata.obsm.get('embedding')
        S = adata.obsm.get('spatial')
        if E is None:
            raise ValueError("adata.obsm['embedding'] not found — run load_embedding_to_ct_spatial_data first.")
        X_pca = self._expr_pca(adata)
        return {
            'silhouette_emb':     self._sil(E, y_pred),
            'silhouette_expr':    self._sil(X_pca, y_pred),   # cross-eval: HGCTSVG clusters in expression PCA
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
