import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq

from matplotlib.colors import ListedColormap
from pathlib import Path

from ..metadata import ProjectMetadata
from ..utils import load_ct_spatial_data, load_decomposed_st_data

def plot_all_ct_decomposition(n_cols = 4, figsize: tuple = (4, 4), cell_color="lime"):
    project_metadata = ProjectMetadata()
    ct_key = project_metadata.ct_key
    cell_types = project_metadata.ct_list
    n_cell_types = project_metadata.n_cell_types

    n_rows = (n_cell_types // n_cols) + (1 if (n_cell_types % n_cols) > 0 else 0)

    width, height = figsize
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(width * n_cols, height * n_rows))
        
    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols][i % n_cols]
        if i < n_cell_types:
            cell_type = cell_types[i]
            
            adata_celltype = load_ct_spatial_data(cell_type)
            sq.pl.spatial_scatter(
                adata_celltype,
                color=ct_key,
                palette=ListedColormap([cell_color]),
                ax=ax,
                title=cell_type,
                legend_loc=None
            )
            
        else:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

def plot_all_ct_domains(n_cols = 4, figsize: tuple = (20, 15)):
    project_metadata = ProjectMetadata()
    type_list = project_metadata.ct_list
    n_cell_types = project_metadata.n_cell_types

    n_rows = (n_cell_types // n_cols) + (1 if (n_cell_types % n_cols) > 0 else 0)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols][i % n_cols]
        if i < n_cell_types:
            cell_type = type_list[i]
            
            adata_celltype = load_ct_spatial_data(cell_type)
            # domain may be saved as plain strings; squidpy needs it categorical
            # to plot as discrete (else it parses labels like '2' as colors)
            adata_celltype.obs['domain'] = adata_celltype.obs['domain'].astype('category')

            sq.pl.spatial_scatter(
                adata_celltype,
                color='domain',
                ax=ax,
                title=cell_type,
                legend_loc=None
            )

            ax.set(xlabel=None, ylabel=None)

        else:
            fig.delaxes(ax)
    
    plt.tight_layout()
    plt.show()

def plot_ct_decomposition(type_list: list = None,
                          figsize: tuple = (4,4),
                          cell_color = "lime",
                          show_title = True,
                          display = False,
                          save = True,
                          transparent = True,
                          file_ext = 'png'):

    project_metadata = ProjectMetadata()
    ct_key = project_metadata.ct_key
    
    if type_list is None:
        type_list = project_metadata.ct_list

    for ct in type_list:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        adata_ct = load_ct_spatial_data(ct)
        sq.pl.spatial_scatter(
            adata_ct,
            color=ct_key,
            palette=ListedColormap([cell_color]),
            title = ct,
            ax=ax,
            legend_loc = None,
            figsize = figsize,
        )

        ax.set(xlabel=None)
        ax.set(ylabel=None)
        
        if not show_title:
            ax.set_title(None)

        plt.tight_layout()
            
        if save:
            save_dir = project_metadata.plot_dir / 'distribution'
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f'distribution_{ct}.{file_ext}', transparent= transparent)

        if not display:
            plt.close(ax.figure)

def plot_ct_domains(type_list: list = None,
                    figsize: tuple = (4, 4),
                    show_title=True,
                    display = False,
                    save=True,
                    transparent=True,
                    file_ext = 'png'):

    project_metadata = ProjectMetadata()
    
    if type_list is None:
        type_list = project_metadata.ct_list

    for ct in type_list:
        adata_ct = load_ct_spatial_data(ct)
        
        ax = sq.pl.spatial_scatter(
            adata_ct,
            color="domain",
            title=ct,
            legend_loc=None,
            figsize = figsize,
            return_ax = True
        )

        ax.set(xlabel=None)
        ax.set(ylabel=None)
        
        if not show_title:
            ax.set_title(None)
        
        plt.tight_layout()
    
        if save:
            save_dir = project_metadata.plot_dir / 'domains'
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f'spatial_domains_{ct}.{file_ext}', transparent=transparent)

        if not display:
            plt.close(ax.figure)

