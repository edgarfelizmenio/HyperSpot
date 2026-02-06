import matplotlib.pyplot as plt
import squidpy as sq

def plot_ct_decomposition(adata, ct_key='cell_type', n_cols = 4, figsize: tuple = (20, 15)):

    cell_types = sorted(adata.obs[ct_key].unique())
    n_cell_types = adata.obs[ct_key].nunique()

    n_rows = (n_cell_types // n_cols) + 1 if (n_cell_types % n_cols) > 0 else 0
    
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