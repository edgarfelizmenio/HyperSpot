import pandas as pd
import pickle
import scanpy as sc

from pathlib import Path

from ..metadata import ProjectMetadata
from ..utils import load_ct_spatial_data

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

def compute_spatial_de_genes(project: Path,
                             cell_type: str,
                             key: str = 'leiden',
                             percentile: int = 0,
                             min_pts: float = 0.8,
                             min_precision: float = 0.1,
                             rank_by: str = 'pvals_adj',
                             use_raw = False
                            ):

    project_metadata = ProjectMetadata()
    
    # load adata for cell type
    adata_ct = load_ct_spatial_data(cell_type)
    
    sc.tl.rank_genes_groups(adata_ct, 
                            groupby=key,
                            method="wilcoxon",
                            pts=True,
                            use_raw=use_raw)
    
    rg = adata_ct.uns['rank_genes_groups']
    groups = list(rg['names'].dtype.names)

    results = {}

    for group in groups:
        df = sc.get.rank_genes_groups_df(
            adata_ct, group=group,
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
            (df['pts'] > 0.8) &
            # select genes expressed more in the region than outside
            ((df['pts'] / df['pts_rest']) > 1) &
            # only select genes that are upregulated
            (df['logfoldchanges'] > 1.5)
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

        threshold = df[rank_by].abs().quantile(percentile)
        top_df = df[df[rank_by].abs() >= threshold]
        results[group] = top_df.sort_values(rank_by)
    
        # df = df.sort_values("combined_score", ascending=False)

        # top_df = df.nsmallest(n_results, "combined_score")
        # results[group] = top_df
        
    return results

def merge_spatial_de_genes(spatial_de_genes_group, 
                           rank_by: str = 'scores'):
    all_genes = {}
    for group in spatial_de_genes_group.keys():
        genes_group = spatial_de_genes_group[group]
        for i in range(len(genes_group)):
            name = genes_group.iloc[i]['names']
            score = genes_group.iloc[i][rank_by]
            if name not in all_genes:
                all_genes[name] = (score, group)
            else:
                current_score, _ = all_genes[name]
                if score > current_score:
                    all_genes[name] = (score, group)

    # sort by
    # num regions (prioritize less regions)
    # average adjusted_pval_per_region
    
    all_genes_df = pd.DataFrame.from_dict(
        all_genes,
        orient="index",
        columns = ['scores', 'region']
    ).sort_values(['region', 'scores'], ascending=[True, False])
    
    return all_genes_df