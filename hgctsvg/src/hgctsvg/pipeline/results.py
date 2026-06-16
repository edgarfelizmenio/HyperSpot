import numpy as np
import pandas as pd
import pickle
import scanpy as sc

from pathlib import Path

from ..metadata import ProjectMetadata
from ..utils import (
    load_ct_spatial_data,
    load_aggr_st_data
)

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

def _run_wilcoxon_per_domain(adata_ct, key, rg):
    """Collect unfiltered Wilcoxon results for all domains of one CT."""
    groups = list(rg['names'].dtype.names)
    domain_dfs = []
    for group in groups:
        df = sc.get.rank_genes_groups_df(adata_ct, group=group)[
            ['names', 'scores', 'pvals', 'logfoldchanges']
        ]
        df['pts']    = df['names'].map(rg['pts'][group])
        df['domain'] = group
        domain_dfs.append(df)
    return pd.concat(domain_dfs, ignore_index=True)


def _ct_safe(ct: str) -> str:
    return ct.replace(' ', '_').replace('+', '_').replace('/', '_')


def compute_ct_svg_scores(
    out_dir: Path,
    key: str = 'domain',
    min_pts: float = 0.8,
    min_logfc: float = 1.0,
):
    """
    Compute per-(gene, cell type) scores, aggregated across domains.

    For each cell type, each (gene, domain) pair is first filtered on
    pts/logfoldchanges. Among the domains passing the filter, the one with
    the max Wilcoxon score is kept per gene; genes with no passing domain
    are dropped entirely. Per-CT BH correction is then applied to the
    surviving genes, and one CSV per cell type is written.
    Columns: CombinedPvals, adj_pval, scores, logfoldchanges, pts, best_domain.
    """
    from statsmodels.stats.multitest import multipletests

    project_metadata = ProjectMetadata()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ct in project_metadata.ct_list:
        adata_ct = load_ct_spatial_data(ct)
        sc.tl.rank_genes_groups(adata_ct, groupby=key, method='wilcoxon',
                                pts=True, use_raw=False)
        rg = adata_ct.uns['rank_genes_groups']
        all_domains = _run_wilcoxon_per_domain(adata_ct, key, rg)

        passing = all_domains[
            (all_domains['pts'] >= min_pts) &
            (all_domains['logfoldchanges'] >= min_logfc)
        ]

        best = passing.loc[passing.groupby('names')['scores'].idxmax()].copy()
        best = best.set_index('names')
        best.index.name = 'gene'
        _, adj_pvals, _, _ = multipletests(best['pvals'].values, method='fdr_bh')
        best['CombinedPvals'] = best['pvals']
        best['adj_pval']      = adj_pvals
        best.rename(columns={'domain': 'best_domain'}, inplace=True)
        best[['CombinedPvals', 'adj_pval', 'scores', 'logfoldchanges',
              'pts', 'best_domain']].to_csv(out_dir / f'{_ct_safe(ct)}.csv')

    print(f"Saved per-CT score files to {out_dir}")


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