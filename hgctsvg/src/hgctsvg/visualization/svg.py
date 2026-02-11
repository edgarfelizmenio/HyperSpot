import matplotlib.pyplot as plt

import scanpy as sc
import squidpy as sq

def plot_region(adata, region_id, ax, region_key="leiden"):
    sq.pl.spatial_scatter(
        adata,
        color=region_key,
        title=f"EC region: {region_id}",
    )

# plot domain top genes
def plot_domain_top_genes(adata, 
                          region, 
                          attr_scores,
                          top_gene_idx, 
                          n_cols = 4, 
                          figsize: tuple = (20, 15), limit=10):
 
    adata_region = adata[adata.obs['leiden'] == str(region)]

    fig, axes = plt.subplots(1, 2, figsize=(20,10))
    # display region
    sq.pl.spatial_scatter(
        adata_region,
        color='leiden',
        title=f"region: {region}",
        ax=axes[0],
        legend_loc=None)

    if limit is None or limit > len(top_gene_idx):
        limit = len(top_gene_idx)
    
    genes = [adata.var_names[idx] for idx in top_gene_idx[:limit]]
    sc.pl.violin(
        adata,
        keys = genes,
        ax = axes[1],
    )

    plt.tight_layout()
    plt.show()

    # plot gene expression
    n_rows = (limit // n_cols) + (1 if (limit % n_cols) > 0 else 0)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20,15))
    axes_flat = axes.flatten()

    for j in range(n_rows * n_cols):
        ax = axes_flat[j]
        if j < limit:
            gene = genes[j]
            gene_idx = top_gene_idx[j]
            
            val = attr_scores[gene_idx]
            vmin = adata[:,gene].X.min()
            vmax = adata[:,gene].X.max()
            
            sq.pl.spatial_scatter(
                adata,
                color=gene,
                ax=axes_flat[j],
                title=f"{gene}: {val:0.4}",
                vmin=vmin,
                vmax=vmax,
                colorbar=False)

        else:
            fig.delaxes(ax)
    
    plt.tight_layout()
    plt.show()
        
    
    
    

def plot_ct_decomposition(adata, ct_key='cell_type', n_cols = 4, figsize: tuple = (20, 15)):

    cell_types = sorted(adata.obs[ct_key].unique())
    n_cell_types = adata.obs[ct_key].nunique()

    n_rows = (n_cell_types // n_cols) + (1 if (n_cell_types % n_cols) > 0 else 0)
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
        
    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols][i % n_cols]
        if i < n_cell_types:
            cell_type = cell_types[i]
            
            adata_celltype = adata[adata.obs[ct_key] == cell_type].copy()
            sq.pl.spatial_scatter(
                adata_celltype,
                color=ct_key,
                ax=ax,
                title=cell_type,
                legend_loc=None
            )
            
        else:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.show()