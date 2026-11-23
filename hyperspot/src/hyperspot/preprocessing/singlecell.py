import pickle
import scanpy as sc
import numpy as np

from ..metadata import ProjectMetadata
from ..utils import save_ct_spatial_data, load_ct_spatial_data

# Make cell type-specific adata objects
def make_ct_spatial_adata(adata_decomp, overwrite=False):
    project_metadata = ProjectMetadata()

    cell_types = project_metadata.ct_list
    cell_type_key = project_metadata.ct_key

    for ct in cell_types:
        out_path = project_metadata.ct_data_dir / f"ST_cell_{ct}.h5ad"
        if not overwrite and out_path.exists():
            print(f"{ct}: {out_path.name} exists, skipping (overwrite=False)")
            continue

        adata_ct = adata_decomp[adata_decomp.obs[cell_type_key] == ct].copy()

        # drop genes that have 0 expression
        pre_filter_genes = adata_ct.n_vars
        sc.pp.filter_genes(adata_ct, min_cells=1)
        post_filter_genes = adata_ct.n_vars
        print(f"{ct} # of genes: {pre_filter_genes} -> {post_filter_genes}")
        # 2026/10/24 - this filtering does not contribute anything
        
        # preprocess cell type-specific data
        adata_ct.raw = adata_ct.copy()
        
        # Spotiphy output is already in CPM so no need to normalize
        # Compute log normalization
        sc.pp.log1p(adata_ct)

        assert adata_ct.X.max() < adata_ct.raw.X.max()
        
        save_ct_spatial_data(adata_ct, ct)

# Perform feature selection on each cell type.
def perform_ct_feature_selection(n_features = 3000, overwrite=False):
    project_metadata = ProjectMetadata()
    ct_feature_filename = project_metadata.project_folder / "ct_features.pkl"
    if not overwrite and ct_feature_filename.exists():
        print(f"{ct_feature_filename.name} exists, skipping feature selection (overwrite=False)")
        return

    cell_types = project_metadata.ct_list

    ct_feats = {}

    for ct in cell_types:
        adata_ct = load_ct_spatial_data(ct)

        feature_genes = compute_feature_genes(adata_ct, n_features)
        # print(f"{ct}:", "dispersions_norm" in adata_ct.var, "feature_gene" in adata_ct.var)
        
        ct_feats[ct] = feature_genes.tolist()
        save_ct_spatial_data(adata_ct, ct)

    # save features for each cell type
    ct_feature_filename = project_metadata.project_folder / "ct_features.pkl"
    with open(ct_feature_filename, "wb") as ctff:
        pickle.dump(ct_feats, ctff)

def compute_feature_genes(adata_ct, n_features=3000):
    # sc.pp.filter_genes(adata_ct, min_cells=1)
    genes = adata_ct.var_names

    sc.pp.highly_variable_genes(adata_ct, flavor="seurat")
    
    # valid_hvg = (adata_ct.var["highly_variable"] == True)
    # hvg_genes = adata_ct.var.index[valid_hvg]

    # hvg_pct = adata_ct.var.loc[hvg_genes, 'dispersions_norm'].sort_values(ascending=False)
    hvg_pct = adata_ct.var['dispersions_norm'].sort_values(ascending=False)
    top_genes = hvg_pct.index[:n_features]
    adata_ct.var['feature_gene'] = False
    adata_ct.var.loc[top_genes, 'feature_gene'] = True
    return top_genes
