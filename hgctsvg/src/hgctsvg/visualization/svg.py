import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from pathlib import Path

from ..metadata import ProjectMetadata

def plot_region(adata, region_id, ax, region_key="leiden"):
    sq.pl.spatial_scatter(
        adata,
        color=region_key,
        title=f"EC region: {region_id}",
    )

def plot_top_domain_de_genes(project: Path, adata, top_genes, regions = list[str], 
                             figsize=(4, 4), n_cols = 4, use_raw=False,
                             save=True, transparent=True, file_ext='png'):
    n_rows = ((len(regions) // n_cols) + (1 if (len(regions) % n_cols) > 0 else 0)) * 2
    width, height = figsize
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(width * n_cols, height * n_rows))
    
    for i in range((n_rows // 2 * n_cols)):
        row = (i // n_cols) * 2
        col = i % n_cols
        ax_region = axes[row][col]
        ax_expr = axes[row + 1][col]
        if i < len(regions):
            region = regions[i]
            adata_region = adata[adata.obs['leiden'] == str(region)]

            gene_id = top_genes[top_genes['region'] == region]['scores'].idxmax()
            gene_name = adata.var.loc[gene_id]['SYMBOL']
            score = top_genes.loc[gene_id]['scores']
            
            if use_raw:
                vals = adata.raw[:,gene_id].X
            else:
                vals = adata[:,gene_id].X
            # remove zeros
            nz = vals[vals > 0]
            vmin = np.percentile(nz, 5)
            vmax = np.percentile(nz, 95)
            
            sq.pl.spatial_scatter(
                adata_region,
                color='leiden',
                title=f"region {region}",
                ax=ax_region,
                legend_loc=None
            )
            sq.pl.spatial_scatter(
                adata,
                color = gene_id,
                ax = ax_expr,
                title=f"{gene_name}: {score:0.4}",
                use_raw=use_raw,
                vmin=vmin,
                vmax=vmax,
                colorbar=False)
            ax_region.axis('off')
            ax_expr.axis('off')
        else:
            fig.delaxes(ax_region)
            fig.delaxes(ax_expr)

    plt.tight_layout()
    
    if save:
        save_dir = project / 'plots'
        save_dir.mkdir(exist_ok=True)
        plt.savefig(project / 'plots' / f'top_de_genes.{file_ext}', transparent=True)
    
    plt.show()


            
def plot_domain_de_genes(adata,
                         region,
                         de_scores,
                         names,
                         n_cols = 4,
                         figsize: tuple = (20, 15), limit=4):
    
    assert limit % n_cols == 0
    adata_region = adata[adata.obs['leiden'] == str(region)]

    fig, axes = plt.subplots(1, 2, figsize=(20,10))
    # display region
    sq.pl.spatial_scatter(
        adata_region,
        color='leiden',
        title=f"region: {region}",
        ax=axes[0],
        legend_loc=None)

    if limit is None or limit > len(names):
        limit = len(names)

    adata_plot = adata.copy()
    adata_plot.var_names = adata.var['SYMBOL']
    adata_plot.var_names_make_unique()
    de_gene_idx = adata.var_names.get_indexer(names[:limit])
    genes = adata_plot.var_names[de_gene_idx]
    
    sc.pl.violin(
        adata_plot,
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
            gene_idx = de_gene_idx[j]
            
            val = de_scores[j]
            vmin = adata_plot.raw[:, gene].X.min()
            vmax = adata_plot.raw[:, gene].X.max()
            
            sq.pl.spatial_scatter(
                adata_plot,
                color=gene,
                ax=axes_flat[j],
                title=f"{genes[j]}: {val:0.4}",
                vmin=vmin,
                vmax=vmax,
                use_raw=True,
                colorbar=False)

        else:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

    del adata_plot

# plot domain top genes
def plot_domain_ct_svgs(adata, 
                        region, 
                        attr_scores,
                        ct_svgs, 
                        n_cols = 4, 
                        figsize: tuple = (20, 15), 
                        limit=12,
                        use_raw = False):
 
    adata_region = adata[adata.obs['leiden'] == str(region)]

    fig, axes = plt.subplots(1, 2, figsize=(20,10))
    # display region
    sq.pl.spatial_scatter(
        adata_region,
        color='leiden',
        title=f"region: {region}",
        ax=axes[0],
        legend_loc=None)

    if limit is None or limit > len(ct_svgs):
        limit = len(ct_svgs)

    adata_plot = adata.copy()
    ct_svgs = ct_svgs[:limit]
    gene_names = adata.var.loc[ct_svgs]['SYMBOL'].tolist()
    
    violin_ax = sc.pl.violin(
        adata_plot,
        keys = ct_svgs,
        ax = axes[1],
        use_raw = use_raw,
        rotation = 45,
        show = False,
    )
    violin_ax.set_xticks(list(range(limit)))
    violin_ax.set_xticklabels(gene_names)
    
    plt.tight_layout()
    plt.show()

    # plot gene expression
    n_rows = (limit // n_cols) + (1 if (limit % n_cols) > 0 else 0)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20,15))
    axes_flat = axes.flatten()

    for j in range(n_rows * n_cols):
        ax = axes_flat[j]
        if j < limit:
            gene = ct_svgs[j]
            
            attr_score = attr_scores[j]

            if use_raw:
                vals = adata_plot.raw[:,gene].X
            else:
                vals = adata_plot[:,gene].X
            # remove zeros
            nz = vals[vals > 0]
            vmin = np.percentile(nz, 5)
            vmax = np.percentile(nz, 95)
            
            sq.pl.spatial_scatter(
                adata_plot,
                color=gene,
                ax=axes_flat[j],
                title=f"{gene_names[j]} ({gene}): {attr_score:0.4}",
                use_raw=use_raw,
                vmin=vmin,
                vmax=vmax,
                size=1,
                colorbar=False)

        else:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
    
    del adata_region, adata_plot    


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

def plot_ct_marker_groups_heatmap(adata, 
                           top_markers: dict, 
                           groupby="leiden",
                           n_genes=10,
                           var_scale=True):

    all_genes = {
        group: top_markers[group].head(n_genes)['names'].tolist() for group in top_markers.keys()
    }

    sc.tl.dendrogram(adata, groupby=groupby)
    sc.pl.heatmap(
        adata,
        var_names = all_genes,
        groupby = groupby,
        swap_axes = False,
        standard_scale = "var" if var_scale else None,
        dendrogram = True,
        show_gene_labels=True
    )