import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq

def plot_ct_domains(project_dir, ct_key='cell_type', n_cols = 4, figsize: tuple = (20, 15)):
    type_list = np.loadtxt(project_dir / "cell_types.txt", dtype=str)
    n_cell_types = len(type_list)

    n_rows = (n_cell_types // n_cols) + (1 if (n_cell_types % n_cols) > 0 else 0)
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols][i % n_cols]
        if i < n_cell_types:
            cell_type = type_list[i]
            
            adata_celltype = sc.read_h5ad(project_dir / "clustering" / f"ST_cell_{cell_type}.h5ad")
        # np_clusters = adata_celltype.obs["leiden"].to_numpy().astype("int")
        
            sq.pl.spatial_scatter(
                adata_celltype,
                # adata_decomp[adata_decomp.obs['cell_type'] == cell_type],
                color="leiden", # or a gene name
                ax=ax,
                title=cell_type,
                legend_loc=None
            )

        else:
            fig.delaxes(ax)
    
    plt.tight_layout()
    plt.show()