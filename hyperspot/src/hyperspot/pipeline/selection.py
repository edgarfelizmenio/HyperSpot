import logging
import numpy as np
import pandas as pd
import scanpy as sc

import scipy.sparse as sp

from ..metadata import ProjectMetadata
from ..utils import (
    load_ct_spatial_data,
    load_aggr_st_data,
    guarded_spot_reps,
)


logger = logging.getLogger(__name__)

# Significance level for calling a (ct)SVG domain (scanpy BH-adjusted p). Used by the
# single detection split (_split_de_table), so the ct-SVG lists and the per-gene table
# (compute_svg_results) call the same genes.
DETECTION_ALPHA = 0.05

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
    pval_adjust: str = 'scanpy',
    pval_thresh: float = None,
    filter_mode: str = 'drop',
    dedup_spots: bool = True,
):
    """Single engine for spatial-DE / (ct)SVG statistics -> flat per-(gene, domain) DataFrame.

    Detection and the per-gene table (compute_spatial_de_genes, compute_svg_results)
    both call it unfiltered through _de_table_all; analysis notebooks and scripts call
    it directly with the flags below.

    method : 'wilcoxon'
        scanpy one-vs-rest Wilcoxon (normal approximation); the only test. Kept as a
        parameter so existing calls passing method='wilcoxon' still work.
    require_specificity : bool
        Add the pts/pts_rest > 1 term (gene detected in a larger fraction of the
        domain than outside). Off in detection: against all other
        cells of the type it is near-redundant once pts >= min_pts and logFC >=
        min_logfc hold (kidney: 0.07% of ct-SVGs at 0.8/1.5). Kept only for the
        sensitivity scripts. pts_rest == 0 counts as specific.
    pval_adjust : 'scanpy'
        scanpy's per-domain BH-adjusted p-value is always reported as `pvals_adj`.
        Kept as a parameter so existing calls passing pval_adjust='scanpy' still work.
    pval_thresh : float or None
        Cutoff applied to the adjusted p-value (None: no p-value filter).
    filter_mode : 'drop' | 'annotate' | 'none'
        How the pts/logfc/specificity/pval mask is applied. 'drop' removes failing
        rows; 'annotate' keeps all rows and adds a `passes_filter` column; 'none'
        keeps all rows with no mask.

    Columns: gene, domain, scores, logfoldchanges, pts, pts_rest, pval, pvals_adj
             [, passes_filter if annotate].
    """
    if method != 'wilcoxon':
        raise ValueError(f"method={method!r}: only 'wilcoxon' is supported")
    if pval_adjust != 'scanpy':
        raise ValueError(f"pval_adjust={pval_adjust!r}: only 'scanpy' is supported")
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
        cols = ['gene', 'domain', 'scores', 'logfoldchanges', 'pts', 'pts_rest', 'pval', 'pvals_adj']
        if filter_mode == 'annotate':
            cols.append('passes_filter')
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})

    # Always pass the explicit list of testable domains (never 'all'): this both
    # excludes singleton domains from being tested (they stay in the one-vs-rest
    # background) and prevents scanpy from touching any 0-cell category.
    sc.tl.rank_genes_groups(adata, groupby=key, groups=list(testable),
                            method='wilcoxon', corr_method='benjamini-hochberg',
                            pts=True, use_raw=False)
    rg = adata.uns['rank_genes_groups']

    df = _run_wilcoxon_per_domain(adata, key, rg)
    df.rename(columns={'names': 'gene', 'pvals': 'pval'}, inplace=True)

    # passing mask: prevalence + fold change (+ optional specificity + optional p)
    mask = (df['pts'] >= min_pts) & (df['logfoldchanges'] >= min_logfc)
    if require_specificity:
        ratio = df['pts'] / df['pts_rest'].replace(0, np.nan)
        mask &= (ratio > 1) | df['pts_rest'].eq(0)
    if pval_thresh is not None:
        mask &= df['pvals_adj'] < pval_thresh

    if filter_mode == 'drop':
        df = df[mask].copy()
    elif filter_mode == 'annotate':
        df = df.copy()
        df['passes_filter'] = mask.values

    cols = ['gene', 'domain', 'scores', 'logfoldchanges', 'pts', 'pts_rest', 'pval', 'pvals_adj']
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

    Wilcoxon, pts/logFC thresholds, and scanpy BH-adjusted p < DETECTION_ALPHA.
    Returns {domain: DataFrame} sorted by `rank_by`; each df carries the columns
    create_svg_list consumes (`names`, `scores`/rank_by, `pvals_adj`). Same single
    DE pass and split as compute_svg_results (which also returns the gene table).
    """
    flat = _de_table_all(adata, key, dedup_spots)
    return _split_de_table(flat, min_pts, min_logfc, rank_by)[0]


GENE_TABLE_COLUMNS = ['CombinedPvals', 'adj_pval', 'scores', 'logfoldchanges',
                      'pts', 'pts_rest', 'best_domain', 'passes_filter']


def _de_table_all(adata, key: str = 'domain', dedup_spots: bool = True):
    """One Wilcoxon pass: every (gene, domain) row, unfiltered, with scanpy BH `pvals_adj`."""
    return _compute_spatial_de_table(
        adata, key=key, method='wilcoxon', min_pts=0.0, min_logfc=-np.inf,
        pval_adjust='scanpy', filter_mode='none', dedup_spots=dedup_spots,
    )


def _split_de_table(flat, min_pts, min_logfc, rank_by='scores', genes=None):
    """Split one unfiltered DE table into detection's {domain: df} and a per-gene table.

    A (gene, domain) row *passes* if pts >= min_pts and logFC >= min_logfc, and is
    *called* if it also has adjusted p < DETECTION_ALPHA.

    domain_svgs : {domain: called rows sorted by rank_by} -- create_svg_list input.
    gene_table  : one row per gene (index), columns GENE_TABLE_COLUMNS. The domain is
        chosen as in detection: the highest-scoring called domain, else the
        highest-scoring passing domain. Genes with no passing domain keep their
        top-scoring domain's statistics but get adj_pval = 1, CombinedPvals = NaN,
        passes_filter = False (a filtered gene: no p-value, never called). With
        `genes`, every gene in it is listed (genes absent from the DE table: all-NaN
        statistics, adj_pval = 1, passes_filter = False). Genes called at
        adj_pval < DETECTION_ALPHA are exactly the genes of domain_svgs.
    """
    passes = (flat['pts'] >= min_pts) & (flat['logfoldchanges'] >= min_logfc)
    called = passes & (flat['pvals_adj'] < DETECTION_ALPHA)

    det = flat[called].rename(columns={'gene': 'names'})
    ascending = rank_by != 'scores'
    domain_svgs = {g: d.sort_values(rank_by, ascending=ascending).reset_index(drop=True)
                   for g, d in det.groupby('domain')}

    p, c = flat[passes], flat[called]
    top_c = c.loc[c.groupby('gene')['scores'].idxmax()]
    top_p = p.loc[p.groupby('gene')['scores'].idxmax()]
    best_pass = pd.concat([top_c, top_p[~top_p['gene'].isin(top_c['gene'])]])
    rest = flat[~flat['gene'].isin(best_pass['gene'])]
    top_rest = (rest.sort_values('scores', ascending=False, kind='mergesort')
                    .drop_duplicates('gene', keep='first'))
    t = pd.concat([best_pass.assign(passes_filter=True),
                   top_rest.assign(passes_filter=False)])
    t = t.set_index('gene').rename(columns={'pval': 'CombinedPvals', 'pvals_adj': 'adj_pval',
                                            'domain': 'best_domain'})
    if genes is not None:
        t = t.reindex(pd.Index(genes).astype(str))
    t['passes_filter'] = t['passes_filter'].fillna(False).astype(bool)
    t.loc[~t['passes_filter'], 'adj_pval'] = 1.0
    t.loc[~t['passes_filter'], 'CombinedPvals'] = np.nan
    t.index.name = 'gene'
    return domain_svgs, t.sort_index()[GENE_TABLE_COLUMNS]


def compute_svg_results(adata,
                        key: str = 'domain',
                        min_pts: float = 0.8,
                        min_logfc: float = 1.5,
                        rank_by: str = 'scores',
                        dedup_spots: bool = True):
    """Detection and per-gene scores from ONE Wilcoxon pass.

    Returns (domain_svgs, gene_table): domain_svgs is detection's {domain: df}
    (create_svg_list(domain_svgs) is the SVG list); gene_table lists every gene of
    `adata` with its best domain, adj_pval / CombinedPvals and passes_filter -- the
    benchmarking format (see _split_de_table). Both use the same rows, so genes
    with adj_pval < DETECTION_ALPHA are exactly the detected genes.
    """
    genes = adata.var_names.astype(str)
    flat = _de_table_all(adata, key, dedup_spots)
    return _split_de_table(flat, min_pts, min_logfc, rank_by, genes)


def compute_ct_svg_results(cell_type: str, **kwargs):
    """compute_svg_results on one cell type's spatial data (ct-SVGs)."""
    return compute_svg_results(load_ct_spatial_data(cell_type), **kwargs)


def compute_tissue_svg_results(**kwargs):
    """compute_svg_results on the spot-level (aggregated) ST data (tissue SVGs)."""
    return compute_svg_results(load_aggr_st_data(), **kwargs)



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
    # The query (ct_spatial_data) is ALREADY CPM(1e6)+log1p from make_ct_spatial_data, normalized over
    # its full gene panel. Put the raw-counts reference on the SAME scale (CPM 1e6 + log1p) over its own
    # full panel, THEN subset both to shared genes. Do NOT re-normalize the query: running normalize_total
    # + log1p on already-log-normalized data double-normalizes it and corrupts the ingest PCA projection.
    sc.pp.normalize_total(ref, target_sum=1e6); sc.pp.log1p(ref)
    common = [g for g in ref.var_names if g in set(q.var_names)]
    ref = ref[:, common].copy(); q = q[:, common].copy()
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