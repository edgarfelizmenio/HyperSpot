import os
import numpy as np
import pandas as pd
import pickle
import scanpy as sc

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..metadata import ProjectMetadata
from ..utils import (
    load_ct_spatial_data,
    load_aggr_st_data
)


def _permutation_worker(ranks, domains, unique_domains, obs_dev, mu, batch_n, seed):
    """Run batch_n permutations and return exceed_count (n_domains, n_genes).

    Module-level so it is picklable by ProcessPoolExecutor.
    """
    rng = np.random.default_rng(seed)
    n_domains = len(unique_domains)
    n_genes = ranks.shape[1]
    exceed_count = np.zeros((n_domains, n_genes), dtype=np.int32)

    perm_labels = np.array([rng.permutation(domains) for _ in range(batch_n)])
    for i, d in enumerate(unique_domains):
        perm_masks = (perm_labels == d).astype(np.float32)   # (batch_n, n_cells)
        perm_W = perm_masks @ ranks                           # (batch_n, n_genes)
        perm_dev = np.abs(perm_W - mu[i])                    # (batch_n, n_genes)
        exceed_count[i] += (perm_dev >= obs_dev[i]).sum(axis=0).astype(np.int32)

    return exceed_count

def save_ct_svgs(cell_type, feat_idx):
    project_metadata = ProjectMetadata()

    ct_svg_groups = get_ct_svg_ids(cell_type, feat_idx)

    ct_svgs = set()
    for group in ct_svg_groups:
        ct_svgs.update(ct_svg_groups[group])

    ct_svgs = pd.Series(sorted(ct_svgs), name="ct-SVGs")

    save_dir = project_metadata.attr_dir / f"cell_type_{cell_type}"            
    
    ct_svgs.to_csv(save_dir / "ct_svgs.csv", index=False)

def get_ct_svg_ids(cell_type, feat_idx: dict[int, list]):
    project_metadata = ProjectMetadata()
    
    with open(project_metadata.project_folder / "ct_features.pkl", "rb") as ctff:
        ct_features = pickle.load(ctff)

    gene_ids = ct_features[cell_type]
    ct_svgs = []
    for group_feat_idx in feat_idx:
        ct_svgs.append([gene_ids[idx] for idx in group_feat_idx])

    return ct_svgs

def compute_domain_svgs(
    key: str = 'domain',
    min_pts: float = 0.8,
    min_logfc: float = 1.5,
    rank_by: str = 'scores',
):
    project_metadata = ProjectMetadata()
    
    # load adata for cell type    
    adata_aggr = load_aggr_st_data()
    
    return compute_spatial_de_genes(
        adata_aggr,
        key,
        min_pts,
        min_logfc,
        rank_by,
    )    
    
def compute_domain_ct_svgs(
    cell_type: str,
    key: str = 'domain',
    min_pts: float = 0.8,
    min_logfc: float = 1.5,
    rank_by: str = 'scores',
):
    project_metadata = ProjectMetadata()
    
    # load adata for cell type    
    adata_ct = load_ct_spatial_data(cell_type)
    
    return compute_spatial_de_genes(
        adata_ct,
        key,
        min_pts,
        min_logfc,
        rank_by,
    )

def compute_spatial_de_genes(adata,
                             key: str = 'domain',
                             min_pts: float = 0.8,
                             min_logfc: float = 1.5,
                             rank_by: str = 'scores',
                            ):

    project_metadata = ProjectMetadata()
    
    sc.tl.rank_genes_groups(adata, 
                            groupby=key,
                            method="wilcoxon",
                            pts=True,
                            use_raw=False)
    
    rg = adata.uns['rank_genes_groups']
    groups = list(rg['names'].dtype.names)

    results = {}

    for group in groups:
        df = sc.get.rank_genes_groups_df(
            adata, group=group,
        )

        # group_idx = groups.index(group)

        df['pts'] = df['names'].map(rg['pts'][group])
        df['pts_rest'] = df['names'].map(rg['pts_rest'][group])
        
        # n_c = (adata_ct.obs[key] == group).sum()
        # n_r = (adata_ct.obs[key] != group).sum()

        # df['recall'] = df['pts']
        # df['precision'] = (
        #     df["pts"] * n_c /
        #     (df["pts"] * n_c + df["pts_rest"] * n_r + 1e-12)
        # )
        
        # df['specificity'] = df['pts'] - df['pts_rest']
        # df['f1'] = (
        #     2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'] + 1e-12)
        # )

        # Filter irrelevant genes, downregulated genes, low coverage genes
        df = df[
            (df['pvals_adj'] < 0.05) &
            # only select genes expressed in at least 80% of the region
            (df['pts'] > min_pts) &
            # select genes expressed more in the region than outside
            ((df['pts'] / df['pts_rest']) > 1) &
            # only select genes that are upregulated
            (df['logfoldchanges'] > min_logfc)
        ]

        # df["consistency_rank"] = df["scores"].rank(pct=True)
        # df["f1_rank"] = df["f1"].rank(pct=True)
        # df["precision_rank"] = df["precision"].rank(pct=True)
        
        # df["consistency+f1"] = (
        #     df["consistency_rank"] + 
        #     df["f1_rank"]
        # ) / 2

        # df["consistency+precision"] = (
        #     df["consistency_rank"] + 
        #     df["precision_rank"]
        # ) / 2

        if rank_by == 'scores':
            results[group] = df.sort_values(rank_by, ascending=False)
        else:
            results[group] = df.sort_values(rank_by)
    
        # df = df.sort_values("combined_score", ascending=False)

        # top_df = df.nsmallest(n_results, "combined_score")
        # results[group] = top_df
        
    return results

def _run_wilcoxon_permutation_per_ct(adata_ct, key, rg,
                                      n_permutations=999, seed=0, n_workers=None):
    """
    Wilcoxon one-vs-rest with permutation-corrected empirical p-values.

    Precomputes gene ranks once, splits permutations into batches, and dispatches
    batches to multiple worker processes via ProcessPoolExecutor. Each worker
    receives only numpy arrays (no AnnData), so pickling is cheap.

    Returns one row per (gene, domain) for all genes (no pts/logfc filter).
    Columns: gene, domain, scores, logfoldchanges, pts, pts_rest, pval.
    """
    from scipy.sparse import issparse
    from scipy.stats import rankdata as _rankdata

    X = adata_ct.X.toarray() if issparse(adata_ct.X) else np.asarray(adata_ct.X)
    X = X.astype(np.float32)
    n_cells, n_genes = X.shape
    gene_names = list(adata_ct.var_names)

    domains = adata_ct.obs[key].values
    unique_domains = np.unique(domains)
    n_domains = len(unique_domains)
    domain_to_idx = {d: i for i, d in enumerate(unique_domains)}
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Precompute per-gene ranks across all cells once — shared across all workers
    ranks = np.apply_along_axis(_rankdata, 0, X).astype(np.float32)  # (n_cells, n_genes)

    # Observed rank sums and expected values under null
    obs_W = np.zeros((n_domains, n_genes), dtype=np.float64)
    n1 = np.zeros(n_domains, dtype=int)
    for i, d in enumerate(unique_domains):
        mask = domains == d
        n1[i] = mask.sum()
        obs_W[i] = ranks[mask].sum(axis=0)

    mu = (n1 * (n_cells + 1) / 2.0)[:, None]   # (n_domains, 1)
    obs_dev = np.abs(obs_W - mu)                 # (n_domains, n_genes)

    # Distribute permutations evenly across workers
    n_workers = n_workers or os.cpu_count() or 1
    n_workers = min(n_workers, n_permutations)
    base, remainder = divmod(n_permutations, n_workers)
    batches = [
        (ranks, domains, unique_domains, obs_dev, mu, base + (1 if i < remainder else 0), seed + i)
        for i in range(n_workers)
    ]

    if n_workers == 1:
        results = [_permutation_worker(*batches[0])]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_permutation_worker, *b) for b in batches]
            results = [f.result() for f in futures]

    exceed_count = sum(results)  # (n_domains, n_genes)

    # Empirical two-sided p-values
    perm_pvals = (exceed_count + 1) / (n_permutations + 1)  # (n_domains, n_genes)

    # Collect Wilcoxon scores, logfc, pts, pts_rest from scanpy results
    all_df = _run_wilcoxon_per_domain(adata_ct, key, rg)

    # Replace theoretical p-values with empirical ones
    all_df['pvals'] = [
        perm_pvals[domain_to_idx[d], gene_to_idx[g]]
        for g, d in zip(all_df['names'], all_df['domain'])
    ]

    all_df.rename(columns={'names': 'gene', 'pvals': 'pval'}, inplace=True)
    return all_df[['gene', 'domain', 'scores', 'logfoldchanges', 'pts', 'pts_rest', 'pval']]


def _run_wilcoxon_per_domain(adata_ct, key, rg):
    """Collect unfiltered Wilcoxon results for all domains of one CT."""
    groups = list(rg['names'].dtype.names)
    domain_dfs = []
    for group in groups:
        df = sc.get.rank_genes_groups_df(adata_ct, group=group)[
            ['names', 'scores', 'pvals', 'logfoldchanges']
        ]
        df['pts']      = df['names'].map(rg['pts'][group])
        df['pts_rest'] = df['names'].map(rg['pts_rest'][group])
        df['domain']   = group
        domain_dfs.append(df)
    return pd.concat(domain_dfs, ignore_index=True)


def _run_kruskal_per_ct(adata_ct, key, rg, min_pts, min_logfc):
    """KW omnibus p-value per gene + Wilcoxon best-domain annotation.

    Uses rank_genes_groups results already stored in rg for best-domain
    annotation (scores, logfc, pts, pts_rest, domain), and replaces the
    per-domain Wilcoxon p-value with a single KW omnibus p-value per gene.
    All genes are returned (no hard filter); passes_filter marks genes that
    meet the pts/logfc criteria.
    """
    from scipy import stats
    from scipy.sparse import issparse

    X = adata_ct.X.toarray() if issparse(adata_ct.X) else adata_ct.X
    domains = adata_ct.obs[key].values
    unique_domains = np.unique(domains)

    kw_pvals = []
    for g_idx in range(X.shape[1]):
        expr = X[:, g_idx]
        if expr.std() == 0:
            kw_pvals.append(1.0)
            continue
        groups = [expr[domains == d] for d in unique_domains]
        try:
            _, pval = stats.kruskal(*groups)
        except ValueError:
            pval = 1.0
        kw_pvals.append(pval)
    kw_pval = dict(zip(adata_ct.var_names, kw_pvals))

    all_domains = _run_wilcoxon_per_domain(adata_ct, key, rg)
    best = all_domains.loc[all_domains.groupby('names')['scores'].idxmax()].copy()
    best.rename(columns={'names': 'gene', 'pvals': 'wilcoxon_pval'}, inplace=True)
    best['pval'] = best['gene'].map(kw_pval)
    best['passes_filter'] = (
        (best['pts'] >= min_pts) & (best['logfoldchanges'] >= min_logfc)
    )
    return best[['gene', 'domain', 'scores', 'logfoldchanges',
                 'pts', 'pts_rest', 'pval', 'passes_filter']]


def _ct_safe(ct: str) -> str:
    return ct.replace(' ', '_').replace('+', '_').replace('/', '_')


def compute_ct_svg_scores(
    out_dir: Path = None,
    key: str = 'domain',
    min_pts: float = 0.8,
    min_logfc: float = 1.0,
    method: str = 'wilcoxon',
    n_permutations: int = 999,
    permutation_seed: int = 0,
    n_workers: int = None,
):
    """
    Score genes for ctSVG detection and save per-CT results.

    method='wilcoxon' (default):
        Runs Wilcoxon one-vs-rest per domain, keeps only genes passing
        pts >= min_pts and logfoldchanges >= min_logfc.
        Output: one row per (gene, domain) pair that passes the filter.
        Columns: gene, domain, scores, logfoldchanges, pts, pts_rest, pval.

    method='kruskal':
        Runs Kruskal-Wallis (omnibus across all domains) for the p-value, and
        Wilcoxon one-vs-rest to annotate the best domain per gene.
        All genes are written; passes_filter marks the pts/logfc criteria.
        Columns: gene, domain, scores, logfoldchanges, pts, pts_rest, pval,
                 passes_filter.

    method='wilcoxon_permutation':
        Runs Wilcoxon one-vs-rest, but replaces the normal-approximation p-value
        with an empirical permutation p-value. Domain labels are shuffled
        n_permutations times; the empirical p-value is the fraction of shuffles
        where the permuted rank sum deviated from the null mean at least as much
        as the observed. Corrects for type-I error inflation caused by
        expression-derived clustering. No pts/logfc filter is applied — all
        (gene, domain) pairs are written.
        Columns: gene, domain, scores, logfoldchanges, pts, pts_rest, pval.

    Output filenames: <ct_safe>_raw.csv  (default dir: ProjectMetadata().domain_markers_dir)
    Run compile_ct_svg_scores() to aggregate and compute adj_pval.
    """
    project_metadata = ProjectMetadata()
    out_dir = Path(out_dir) if out_dir is not None else project_metadata.domain_markers_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for ct in project_metadata.ct_list:
        adata_ct = load_ct_spatial_data(ct)
        sc.tl.rank_genes_groups(adata_ct, groupby=key, method='wilcoxon',
                                pts=True, use_raw=False)
        rg = adata_ct.uns['rank_genes_groups']

        if method == 'kruskal':
            result = _run_kruskal_per_ct(adata_ct, key, rg, min_pts, min_logfc)
            result.to_csv(out_dir / f'{_ct_safe(ct)}_raw.csv', index=False)
        elif method == 'wilcoxon_permutation':
            result = _run_wilcoxon_permutation_per_ct(
                adata_ct, key, rg,
                n_permutations=n_permutations,
                seed=permutation_seed,
                n_workers=n_workers,
            )
            result.to_csv(out_dir / f'{_ct_safe(ct)}_raw.csv', index=False)
        else:
            all_domains = _run_wilcoxon_per_domain(adata_ct, key, rg)
            passing = all_domains[
                (all_domains['pts']           >= min_pts) &
                (all_domains['logfoldchanges'] >= min_logfc)
            ].copy()
            passing.rename(columns={'names': 'gene', 'pvals': 'pval'}, inplace=True)
            passing[['gene', 'domain', 'scores', 'logfoldchanges',
                     'pts', 'pts_rest', 'pval']].to_csv(
                out_dir / f'{_ct_safe(ct)}_raw.csv', index=False
            )

    print(f"Saved per-CT per-domain scores to {out_dir}")


def compile_ct_svg_scores(
    in_dir: Path = None,
    out_dir: Path = None,
):
    """
    Aggregate per-CT per-domain scores and compute adj_pval for benchmarking.

    Reads <ct_safe>_raw.csv files written by compute_ct_svg_scores(), then
    for each cell type:
      1. Selects the best domain per gene (max Wilcoxon score).
      2. Applies per-CT BH correction on the raw p-values of surviving genes.
      3. Writes one benchmarking CSV per cell type.

    Default in_dir and out_dir: ProjectMetadata().domain_markers_dir
    Output filenames: <ct_safe>.csv
    Output columns per gene: CombinedPvals, adj_pval, scores, logfoldchanges,
                              pts, pts_rest, best_domain.
    """
    from statsmodels.stats.multitest import multipletests

    project_metadata = ProjectMetadata()
    in_dir  = Path(in_dir)  if in_dir  is not None else project_metadata.domain_markers_dir
    out_dir = Path(out_dir) if out_dir is not None else project_metadata.domain_markers_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for ct in project_metadata.ct_list:
        raw_csv = in_dir / f'{_ct_safe(ct)}_raw.csv'
        if not raw_csv.exists():
            continue
        df = pd.read_csv(raw_csv)

        best = df.loc[df.groupby('gene')['scores'].idxmax()].copy()
        best = best.set_index('gene')
        if best.empty:
            out_cols = ['CombinedPvals', 'adj_pval', 'scores', 'logfoldchanges',
                        'pts', 'pts_rest', 'best_domain']
            if 'passes_filter' in df.columns:
                out_cols.append('passes_filter')
            pd.DataFrame(columns=out_cols).to_csv(out_dir / f'{_ct_safe(ct)}.csv')
            continue
        _, adj_pvals, _, _ = multipletests(best['pval'].values, method='fdr_bh')
        best['CombinedPvals'] = best['pval']
        best['adj_pval']      = adj_pvals
        if 'passes_filter' in best.columns:
            best.loc[~best['passes_filter'].astype(bool), 'adj_pval'] = 1.0
        best.rename(columns={'domain': 'best_domain'}, inplace=True)
        out_cols = ['CombinedPvals', 'adj_pval', 'scores', 'logfoldchanges',
                    'pts', 'pts_rest', 'best_domain']
        if 'passes_filter' in best.columns:
            out_cols.append('passes_filter')
        best[out_cols].to_csv(out_dir / f'{_ct_safe(ct)}.csv')

    print(f"Compiled per-CT SVG scores to {out_dir}")


def compute_ct_domain_svg_scores(
    out_dir: Path,
    key: str = 'domain',
):
    """
    Compute per-(gene, cell type, domain) scores — unfiltered, one file per domain.

    Same Wilcoxon as compute_ct_svg_scores but does NOT aggregate across domains.
    A gene can appear in multiple domain files. BH correction applied per domain.
    Writes <ct_safe>_domain_<domain>.csv per (CT, domain).
    Columns: CombinedPvals, adj_pval, scores, logfoldchanges, pts.

    Use filter_svg_scores() to apply score/logfc/pts thresholds afterwards.
    """
    from statsmodels.stats.multitest import multipletests

    project_metadata = ProjectMetadata()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ct in project_metadata.ct_list:
        adata_ct = load_ct_spatial_data(ct)
        sc.tl.rank_genes_groups(adata_ct, groupby=key, method='wilcoxon',
                                pts=True, use_raw=True)
        rg = adata_ct.uns['rank_genes_groups']
        all_domains = _run_wilcoxon_per_domain(adata_ct, key, rg)

        for domain, grp in all_domains.groupby('domain'):
            grp = grp.set_index('names').copy()
            grp.index.name = 'gene'
            _, adj_pvals, _, _ = multipletests(grp['pvals'].values, method='fdr_bh')
            grp['CombinedPvals'] = grp['pvals']
            grp['adj_pval']      = adj_pvals
            grp[['CombinedPvals', 'adj_pval', 'scores',
                  'logfoldchanges', 'pts']].to_csv(
                out_dir / f'{_ct_safe(ct)}_domain_{domain}.csv'
            )

    print(f"Saved per-(CT, domain) score files to {out_dir}")


def filter_svg_scores(
    in_dir: Path,
    out_dir: Path,
    # min_score: float = 0.0,
    # min_logfc: float = 0.0,
    min_pts: float = 0.0,
    min_logfc: float = 0.0,
):
    """
    Filter CSV files produced by compute_ct_svg_scores or compute_ct_domain_svg_scores.

    Reads all *.csv files in in_dir, applies score/logfc/pts thresholds,
    and writes filtered versions to out_dir (same filenames).
    """
    project_metadata = ProjectMetadata()
    in_dir  = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ct in project_metadata.ct_list:
        csv = in_dir / f'{_ct_safe(ct)}.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv, index_col=0)
        mask = (
            # (df['scores']          >= min_score) &
            (df['pts']             >= min_pts) &
            (df['logfoldchanges']  >= min_logfc)
        )
        df[mask].to_csv(out_dir / csv.name)

    print(f"Filtered CSVs written to {out_dir}")

def create_svg_list(
    domain_svgs, 
    rank_by: str = 'scores'
):
    
    all_genes = {}
    for group in domain_svgs.keys():
        genes_group = domain_svgs[group]
        for i in range(len(genes_group)):
            name = genes_group.iloc[i]['names']
            score = genes_group.iloc[i][rank_by]
            pvals_adj = genes_group.iloc[i]['pvals_adj']
            if name not in all_genes:
                all_genes[name] = (score, pvals_adj, group)
            else:
                current_score, _, _ = all_genes[name]
                if score > current_score:
                    all_genes[name] = (score, pvals_adj, group)

    # sort by
    # num regions (prioritize less regions)
    # average adjusted_pval_per_region
    
    all_genes_df = pd.DataFrame.from_dict(
        all_genes,
        orient="index",
        columns = ['scores', 'adjusted p-vals', 'region']
    ).sort_values(['region', 'scores'], ascending=[True, False])
    
    return all_genes_df