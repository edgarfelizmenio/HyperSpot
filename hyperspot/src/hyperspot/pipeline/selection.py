import os
import numpy as np
import pandas as pd
import pickle
import scanpy as sc

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import scipy.sparse as sp

from ..metadata import ProjectMetadata
from ..utils import (
    load_ct_spatial_data,
    load_aggr_st_data,
    guarded_spot_reps,
)


def dedup_by_spot(adata, tol=1e-3, sample=100):
    """Collapse same-spot cells to one representative for spatial DE.

    Same-spot same-CT cells are EXACT deconvolution duplicates: Spotiphy assigns
    one per-type profile per spot, so every cell of a type in a spot has identical
    expression. Counting each replicate inflates the Wilcoxon sample size AND the
    `pts` (fraction expressing), making genes look artificially significant and
    over-weighting dense spots. Deduping to unique spots makes each an independent
    observation -- lossless *because* the cells are identical.

    Guarded (shares `guarded_spot_reps` with the clustering dedup): only dedup when
    same-spot cells really are (near-)identical; on data with genuine per-cell
    variation return adata unchanged so the standard per-cell test is used. Per-spot
    ST data (unique spots) is a natural no-op. Expression is wide/sparse -> the
    guard checks a sample of multi-cell spots.
    """
    if 'spot_name' not in adata.obs or adata.n_obs == 0:
        return adata
    rep_idx, _ = guarded_spot_reps(adata.X, adata.obs['spot_name'].to_numpy(),
                                   dedup=True, tol=tol, sample=sample)
    if len(rep_idx) == adata.n_obs:
        return adata                                     # no dedup (no dups, or genuine variation)
    return adata[rep_idx].copy()


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
    dedup_spots: bool = True,
):
    project_metadata = ProjectMetadata()

    # ST-level (per-spot aggregated) data -> unique spots -> dedup is a no-op
    adata_aggr = load_aggr_st_data()

    return compute_spatial_de_genes(
        adata_aggr,
        key,
        min_pts,
        min_logfc,
        rank_by,
        dedup_spots=dedup_spots,
    )
    
def compute_domain_ct_svgs(
    cell_type: str,
    key: str = 'domain',
    min_pts: float = 0.8,
    min_logfc: float = 1.5,
    rank_by: str = 'scores',
    dedup_spots: bool = True,
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
        dedup_spots=dedup_spots,
    )

def _compute_spatial_de_table(
    adata,
    key: str = 'domain',
    method: str = 'wilcoxon',
    min_pts: float = 0.8,
    min_logfc: float = 1.5,
    require_specificity: bool = False,
    pval_adjust: str = 'none',
    pval_thresh: float = None,
    filter_mode: str = 'drop',
    best_domain_per_gene: bool = False,
    dedup_spots: bool = True,
    n_permutations: int = 999,
    permutation_seed: int = 0,
    n_workers: int = None,
):
    """Single engine for spatial-DE / (ct)SVG statistics -> flat per-(gene, domain) DataFrame.

    Shared by the tissue-wide SVG path (compute_domain_svgs), the per-cell-type
    ctSVG detection path (compute_domain_ct_svgs) -- both via compute_spatial_de_genes
    -- and the scoring path (compute_ct_svg_scores); these differ only in the flags
    below.

    method : 'wilcoxon' | 'kruskal' | 'wilcoxon_permutation'
        Which p-value to report in `pval`. 'wilcoxon' = scanpy one-vs-rest normal
        approximation; 'kruskal' = KW omnibus per gene (annotated with the best
        Wilcoxon domain, so inherently one row per gene); 'wilcoxon_permutation' =
        empirical permutation p-value.
    require_specificity : bool
        Add the pts/pts_rest > 1 term (gene detected in a larger fraction of the
        domain than outside). Enabled by the detection wrapper, disabled by scoring
        -- with pts >= min_pts already enforced it is near-redundant, so scoring
        leaves it off. pts_rest == 0 counts as specific.
    pval_adjust : 'scanpy' | 'none'
        'scanpy' surfaces scanpy's BH-adjusted p-value as `pvals_adj` and lets
        `pval_thresh` filter on it (only valid for method='wilcoxon'). 'none' reports
        no adjusted p-value and applies no p-value filter (scoring defers BH to
        compile_ct_svg_scores).
    pval_thresh : float or None
        Cutoff applied to the adjusted p-value; requires pval_adjust='scanpy'.
    filter_mode : 'drop' | 'annotate' | 'none'
        How the pts/logfc/specificity/pval mask is applied. 'drop' removes failing
        rows (detection, wilcoxon scoring); 'annotate' keeps all rows and adds a
        `passes_filter` column (kruskal scoring); 'none' keeps all rows with no mask
        (permutation scoring, where the empirical p-value stands alone).
    best_domain_per_gene : bool
        Collapse to one row per gene (its max-Wilcoxon-score domain).

    Columns: gene, domain, scores, logfoldchanges, pts, pts_rest, pval
             [, pvals_adj if pval_adjust='scanpy'] [, passes_filter if annotate].
    """
    # dedup_spots=True (default, Spotiphy): same-spot same-CT cells are exact
    # deconvolution duplicates -> collapse to unique spots so the Wilcoxon test
    # (and pts) treats each spot as one observation, not N identical replicates.
    # Internally guarded, so it is a no-op if cells actually vary. Set False for
    # benchmarks on real per-cell expression to force the standard per-cell test.
    if dedup_spots:
        adata = dedup_by_spot(adata)

    # Robustness for sparse cell types (e.g. pDC, mast): after dedup a domain can
    # hold a single observation, which Wilcoxon cannot test ("groups 0 ... only
    # contain one sample"). Restrict the tested groups to domains with >= 2
    # observations (singleton domains stay in the one-vs-rest background). If fewer
    # than two domains remain or none is testable there is no ct-SVG structure to
    # detect, so return an empty table and let the caller skip this cell type
    # instead of crashing the whole run.
    # Drop stale/unused cluster labels first: a categorical carrying 0-cell
    # categories makes scanpy try to test empty groups ("only contain one sample"
    # crash, or Mean-of-empty-slice / divide-by-zero warnings).
    adata.obs[key] = adata.obs[key].astype('category').cat.remove_unused_categories()
    counts = adata.obs[key].value_counts()
    counts = counts[counts > 0]
    testable = [g for g, c in counts.items() if c >= 2]
    if len(testable) == 0 or len(counts) < 2:
        import warnings
        warnings.warn(
            f"_compute_spatial_de_table: skipping DE on '{key}' -- need >=2 domains "
            f"with >=2 observations (have {len(counts)} domain(s), sizes "
            f"{counts.to_dict()}); returning no genes (sparse cell type).")
        cols = ['gene', 'domain', 'scores', 'logfoldchanges', 'pts', 'pts_rest', 'pval']
        if pval_adjust == 'scanpy':
            cols.append('pvals_adj')
        if filter_mode == 'annotate':
            cols.append('passes_filter')
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})

    # Always pass the explicit list of testable domains (never 'all'): this both
    # excludes singleton domains from being tested (they stay in the one-vs-rest
    # background) and prevents scanpy from touching any 0-cell category.
    sc.tl.rank_genes_groups(adata, groupby=key, groups=list(testable),
                            method='wilcoxon', pts=True, use_raw=False)
    rg = adata.uns['rank_genes_groups']

    if method == 'kruskal':
        # already one row per gene (KW omnibus); recompute passes_filter uniformly
        df = _run_kruskal_per_ct(adata, key, rg, min_pts, min_logfc)
        df = df.drop(columns=['passes_filter'])
    elif method == 'wilcoxon_permutation':
        df = _run_wilcoxon_permutation_per_ct(
            adata, key, rg, n_permutations=n_permutations,
            seed=permutation_seed, n_workers=n_workers)
    else:  # wilcoxon
        df = _run_wilcoxon_per_domain(adata, key, rg)
        df.rename(columns={'names': 'gene', 'pvals': 'pval'}, inplace=True)

    if pval_adjust == 'scanpy':
        if 'pvals_adj' not in df.columns:
            raise ValueError(
                "pval_adjust='scanpy' is only available for method='wilcoxon'")
        # keep pvals_adj as-is (already carried through); nothing else to do

    # passing mask: prevalence + fold change (+ optional specificity + optional p)
    mask = (df['pts'] >= min_pts) & (df['logfoldchanges'] >= min_logfc)
    if require_specificity:
        ratio = df['pts'] / df['pts_rest'].replace(0, np.nan)
        mask &= (ratio > 1) | df['pts_rest'].eq(0)
    if pval_adjust == 'scanpy' and pval_thresh is not None:
        mask &= df['pvals_adj'] < pval_thresh

    if filter_mode == 'drop':
        df = df[mask].copy()
    elif filter_mode == 'annotate':
        df = df.copy()
        df['passes_filter'] = mask.values

    if best_domain_per_gene:
        df = df.loc[df.groupby('gene')['scores'].idxmax()]

    cols = ['gene', 'domain', 'scores', 'logfoldchanges', 'pts', 'pts_rest', 'pval']
    if pval_adjust == 'scanpy' and 'pvals_adj' in df.columns:
        cols.append('pvals_adj')
    if 'passes_filter' in df.columns:
        cols.append('passes_filter')
    return df[cols].reset_index(drop=True)


def compute_spatial_de_genes(adata,
                             key: str = 'domain',
                             min_pts: float = 0.8,
                             min_logfc: float = 1.5,
                             rank_by: str = 'scores',
                             dedup_spots: bool = True,
                            ):
    """Detection path: filtered spatial DE genes per domain, as {domain: df}.

    Thin wrapper over _compute_spatial_de_table with the detection configuration:
    Wilcoxon, specificity on (pts/pts_rest > 1), and scanpy BH-adjusted p < 0.05.
    Returns {domain: DataFrame} sorted by `rank_by`; each df carries the columns
    create_svg_list consumes (`names`, `scores`/rank_by, `pvals_adj`).
    """
    flat = _compute_spatial_de_table(
        adata,
        key=key,
        method='wilcoxon',
        min_pts=min_pts,
        min_logfc=min_logfc,
        require_specificity=True,
        pval_adjust='scanpy',
        pval_thresh=0.05,
        filter_mode='drop',
        dedup_spots=dedup_spots,
    )
    flat = flat.rename(columns={'gene': 'names'})

    results = {}
    ascending = rank_by != 'scores'
    for group, grp in flat.groupby('domain'):
        results[group] = grp.sort_values(rank_by, ascending=ascending).reset_index(drop=True)
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
            ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']
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
    pval_adjust: str = 'scanpy',
    n_permutations: int = 999,
    permutation_seed: int = 0,
    n_workers: int = None,
    dedup_spots: bool = True,
    best_domain_per_gene: bool = False,
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

    dedup_spots (default True): collapse same-spot same-CT duplicate cells to one
        representative before scoring, so the Wilcoxon/KW test and pts treat each spot
        as one observation instead of N identical replicates (aligns with the detection
        path compute_domain_ct_svgs). Guarded -> no-op if cells genuinely vary. Set
        False to score on all cells (real per-cell expression benchmarks).

    best_domain_per_gene (default False): output granularity. False keeps the full
        gene list (one row per passing gene-domain); True collapses to one row per
        gene, its max-Wilcoxon-score domain. (kruskal already returns best-domain-
        per-gene -- its p-value is a single omnibus per gene -- so the flag is a no-op
        there.)

    pval_adjust (default 'scanpy'): how compile_ct_svg_scores derives adj_pval.
        'scanpy' carries scanpy's inline BH-adjusted p-value (pvals_adj) through to
        the raw CSV, aligning scoring with the detection path (compute_domain_ct_svgs),
        which uses the same adjustment. Only defined for method='wilcoxon'; kruskal /
        wilcoxon_permutation have no scanpy pvals_adj, so they fall back to 'none'
        (compile applies its own best-domain BH). 'none' forces compile-step BH for
        any method.

    Output filenames: <ct_safe>_raw.csv  (default dir: ProjectMetadata().domain_markers_dir)
    Run compile_ct_svg_scores() to aggregate and finalize adj_pval.
    """
    project_metadata = ProjectMetadata()
    out_dir = Path(out_dir) if out_dir is not None else project_metadata.domain_markers_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # scoring path: same engine as detection, but specificity off and no inline
    # p-value filter (BH is deferred to compile_ct_svg_scores). filter_mode keeps
    # each method's granularity -- wilcoxon drops non-passing genes; kruskal keeps
    # all and annotates passes_filter; permutation keeps all (empirical p stands).
    filter_mode = {'kruskal': 'annotate', 'wilcoxon_permutation': 'none'}.get(method, 'drop')
    # scanpy inline BH (pvals_adj) is only defined for the Wilcoxon test; for the
    # other statistics there is no scanpy-adjusted p-value, so defer to compile-BH.
    eff_adjust = pval_adjust if method == 'wilcoxon' else 'none'

    for ct in project_metadata.ct_list:
        adata_ct = load_ct_spatial_data(ct)
        result = _compute_spatial_de_table(
            adata_ct,
            key=key,
            method=method,
            min_pts=min_pts,
            min_logfc=min_logfc,
            require_specificity=False,
            pval_adjust=eff_adjust,
            filter_mode=filter_mode,
            best_domain_per_gene=best_domain_per_gene,
            dedup_spots=dedup_spots,
            n_permutations=n_permutations,
            permutation_seed=permutation_seed,
            n_workers=n_workers,
        )
        result.to_csv(out_dir / f'{_ct_safe(ct)}_raw.csv', index=False)

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
      2. Finalizes adj_pval: if the raw file carries scanpy's pvals_adj
         (pval_adjust='scanpy', the default -- aligned with detection), the best
         domain's scanpy-adjusted p-value is used directly; otherwise a per-CT BH
         correction is applied to the raw p-values of surviving genes.
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
        best['CombinedPvals'] = best['pval']
        if 'pvals_adj' in best.columns:
            # scanpy inline BH carried through from compute_ct_svg_scores(pval_adjust=
            # 'scanpy'): use the best domain's scanpy-adjusted p-value directly, so
            # scoring shares the detection path's adjustment (no compile-step BH).
            best['adj_pval'] = best['pvals_adj'].values
        else:
            _, adj_pvals, _, _ = multipletests(best['pval'].values, method='fdr_bh')
            best['adj_pval'] = adj_pvals
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


def compute_ct_metagenes(
    cell_type: str,
    k: int = 3,
    genes: list = None,
    min_pts: float = 0.8,
    min_logfc: float = 1.5,
    n_top: int = 40,
    random_state: int = 0,
    max_iter: int = 800,
):
    """Meta-gene decomposition of a cell type's spatial program (CELINA-style).

    A ct-SVG *domain* is a hard spatial partition; a *metagene* is a soft gene module
    -- a group of genes that co-vary in space regardless of domain boundaries. CELINA
    (Shang et al. 2025) reconstructs kidney RCC metagenes (PT / EMT / IFN-gamma) by
    clustering ct-SVGs on their spatial pattern. This does the equivalent with
    non-negative matrix factorization on the (cell-type-attributed) spot x gene matrix
    restricted to that cell type's ct-SVGs: X (spots x genes) ~= W (spots x k) @ H
    (k x genes). Each of the k rows of H is a metagene (gene loadings); each column of
    W is that metagene's spatial intensity across spots.

    Parameters
    ----------
    cell_type : which cell type's ct-spatial adata to decompose (load_ct_spatial_data).
    k         : number of metagenes (CELINA used 3 for RCC).
    genes     : gene set to factorize; default = the union of this cell type's per-domain
                ct-SVGs (compute_domain_ct_svgs), i.e. the spatially variable program only.
    n_top     : how many top-loading genes to list per metagene.

    Returns a dict:
      loadings    -- DataFrame (genes x k) of H.T, gene loadings per metagene.
      spot_scores -- DataFrame (spots x k) of W, per-spot metagene intensity (index = obs).
      top_genes   -- {j: [gene, ...]} top-n_top genes per metagene by loading.
      genes       -- the gene set actually used.
      recon_err   -- NMF reconstruction error.
    """
    from sklearn.decomposition import NMF

    adata_ct = load_ct_spatial_data(cell_type)
    if genes is None:
        dom = compute_domain_ct_svgs(cell_type, min_pts=min_pts, min_logfc=min_logfc)
        genes = sorted({
            g for df in dom.values()
            for g in (df['names'] if 'names' in df.columns else df.index)
        })
    genes = [g for g in genes if g in adata_ct.var_names]
    if len(genes) < k:
        raise ValueError(f"only {len(genes)} of the requested genes are in "
                         f"{cell_type}'s var; need >= k={k}.")

    A = adata_ct[:, genes]
    X = A.X
    X = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
    X = np.maximum(X, 0.0)  # NMF requires non-negative input

    nmf = NMF(n_components=k, init='nndsvda', random_state=random_state,
              max_iter=max_iter)
    W = nmf.fit_transform(X)
    H = nmf.components_

    genes = np.asarray(genes)
    cols = [f"metagene_{j}" for j in range(k)]
    loadings = pd.DataFrame(H.T, index=genes, columns=cols)
    spot_scores = pd.DataFrame(W, index=A.obs_names, columns=cols)
    top_genes = {
        j: list(genes[np.argsort(H[j])[::-1][:n_top]]) for j in range(k)
    }
    return {
        "loadings": loadings,
        "spot_scores": spot_scores,
        "top_genes": top_genes,
        "genes": list(genes),
        "recon_err": float(nmf.reconstruction_err_),
    }


def compute_domain_pathways(
    cell_type: str,
    sources=("GO:BP", "REAC", "KEGG"),
    min_genes: int = 15,
    min_pts: float = 0.8,
    min_logfc: float = 1.5,
    organism: str = "hsapiens",
):
    """Functional enrichment of each spatial DOMAIN of a cell type -> a domain x pathway table.

    A ct-SVG domain is a spatial partition of a cell type; its marker ct-SVGs
    (compute_domain_ct_svgs) define a gene program. Running g:Profiler (sc.queries.enrich)
    per domain turns each domain into a set of enriched pathways -- i.e. HyperSpot resolves a
    cell type's spatial domains into distinct FUNCTIONAL units. This is an output the
    regression-based ct-SVG methods (CELINA/STANCE) cannot produce: they yield genes-per-cell-type
    with no spatial domains, hence no per-domain pathways.

    Requires per-cell-type domains (compute_celltype_spatial_domains) and network access
    (g:Profiler API). Domains with < `min_genes` ct-SVGs are skipped.

    Returns a long DataFrame: [domain, n_genes, source, native, pathway, p_value, term_size].
    """
    import scanpy as sc

    dom = compute_domain_ct_svgs(cell_type, min_pts=min_pts, min_logfc=min_logfc)
    rows = []
    for d, df in dom.items():
        genes = list(df["names"]) if "names" in df.columns else list(df.index)
        if len(genes) < min_genes:
            continue
        res = sc.queries.enrich(
            genes, org=organism,
            gprofiler_kwargs={"sources": list(sources), "no_evidences": True})
        if not len(res):
            continue
        for _, r in res.iterrows():
            rows.append((str(d), len(genes), r["source"], r["native"],
                         r["name"], r["p_value"], r["term_size"]))
    return pd.DataFrame(rows, columns=["domain", "n_genes", "source", "native",
                                       "pathway", "p_value", "term_size"])


def map_domains_to_subtypes(adata_ref, cell_type: str, subtype_col: str = "annotation",
                            broad_col: str = "broad_type", min_cells: int = 80,
                            drop=("Unknown", "Low quality"), max_per_subtype: int = 500,
                            n_comps: int = 30, seed: int = 0):
    """Validate that a cell type's spatial DOMAINS correspond to its scRNA SUBTYPES.

    Transfers subtype labels from an annotated scRNA reference onto the decomposed ST cells of
    `cell_type` (scanpy `ingest`: kNN label transfer in the reference's PCA space), then compares
    the transferred subtypes to HyperSpot's domains. Domains are a *finer, spatial* partition, so
    expect high homogeneity / low completeness for the raw domains; merging domains by their
    dominant subtype should raise completeness -- evidence the domains are a coherent refinement of
    real cell states.

    adata_ref : scRNA reference AnnData with `broad_col` (cell type) and `subtype_col` (subtype).
    Well-populated subtypes only (>= `min_cells`, excluding `drop`); the reference is subsampled to
    <= `max_per_subtype` cells per subtype for speed/balance.

    Returns a dict:
      crosstab          -- DataFrame (domain x transferred-subtype) counts.
      domain_to_subtype -- {domain: dominant subtype} (the merge map).
      metrics           -- DataFrame [homogeneity, completeness, v_measure, ARI] for rows
                           'domains' and 'merged_by_subtype'.
      subtypes          -- the subtypes used.
    """
    import numpy as _np
    from sklearn.metrics import (homogeneity_completeness_v_measure as _hcv,
                                 adjusted_rand_score as _ari)

    rng = _np.random.default_rng(seed)
    ref = adata_ref[adata_ref.obs[broad_col] == cell_type].copy()
    vc = ref.obs[subtype_col].value_counts()
    keep = [s for s, n in vc.items() if n >= min_cells and s not in drop]
    ref = ref[ref.obs[subtype_col].isin(keep)].copy()
    # subsample per subtype for speed/balance
    idx = []
    for s in keep:
        cells = _np.where(ref.obs[subtype_col].values == s)[0]
        idx += list(rng.choice(cells, size=min(max_per_subtype, len(cells)), replace=False))
    ref = ref[sorted(idx)].copy()
    ref.obs["subtype"] = ref.obs[subtype_col].astype(str)

    q = load_ct_spatial_data(cell_type)
    q.obs["domain"] = q.obs["domain"].astype(str)
    common = [g for g in ref.var_names if g in set(q.var_names)]
    ref = ref[:, common].copy(); q = q[:, common].copy()
    for a in (ref, q):
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.pca(ref, n_comps=min(n_comps, len(common) - 1, ref.n_obs - 1))
    sc.pp.neighbors(ref); sc.tl.umap(ref)
    sc.tl.ingest(q, ref, obs="subtype")

    dom = q.obs["domain"].astype(str).values
    sub = q.obs["subtype"].astype(str).values
    ct = pd.crosstab(pd.Series(dom, name="domain"), pd.Series(sub, name="subtype"))
    dommap = {d: ct.loc[d].idxmax() for d in ct.index}
    merged = _np.array([dommap[d] for d in dom])
    h, c, v = _hcv(sub, dom); hm, cm, vm = _hcv(sub, merged)
    metrics = pd.DataFrame(
        {"homogeneity": [h, hm], "completeness": [c, cm], "v_measure": [v, vm],
         "ARI": [_ari(sub, dom), _ari(sub, merged)]},
        index=["domains", "merged_by_subtype"])
    return {"crosstab": ct, "domain_to_subtype": dommap, "metrics": metrics,
            "subtypes": keep, "query": q}


def domain_signature_pathways(pathways_df, per_domain: int = 3,
                              term_size_max: int = 500, min_score: float = 2.0):
    """Pick each domain's SIGNATURE pathways from a compute_domain_pathways() table.

    Ranks pathways by a domain-specificity score = (this domain's -log10 p) minus the best
    -log10 p over the *other* domains, so a pathway enriched everywhere (e.g. OXPHOS across all
    RCC domains) is demoted and each domain surfaces what is distinctive to it.

    Returns (signatures, score_matrix):
      signatures   -- {domain: [(pathway, -log10 p), ...]} top `per_domain` domain-specific pathways.
      score_matrix -- DataFrame (pathway x domain) of -log10 p (0 where absent), for heatmaps.
    """
    d = pathways_df[pathways_df["term_size"] <= term_size_max].copy()
    d["score"] = -np.log10(d["p_value"])
    M = d.pivot_table(index="pathway", columns="domain", values="score", fill_value=0.0)
    signatures = {}
    for dom in M.columns:
        others = M.drop(columns=dom).max(axis=1)
        spec = (M[dom] - others).sort_values(ascending=False)
        signatures[dom] = [(p, round(float(M.loc[p, dom]), 1))
                           for p in spec.head(per_domain).index if M.loc[p, dom] >= min_score]
    return signatures, M