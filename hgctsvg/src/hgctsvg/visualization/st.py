import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

from ..metadata import ProjectMetadata
from ..utils import load_orig_st_data

def plot_st_gene_counts(figsize: tuple = (4, 4),
                        show_title: bool = True,
                        display = False,
                        save = True,
                        transparent = True,
                        file_ext = 'png'):

    project_metadata = ProjectMetadata()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    adata_st = load_orig_st_data()
    sq.pl.spatial_scatter(
        adata_st,
        color="n_genes_by_counts", # or a gene name
        title = 'ST Data',
        ax = ax,
        colorbar = False,
        figsize = figsize,
    )
    
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    
    if not show_title:
        ax.set_title(None)
    
    plt.tight_layout()

    if save:
        plt.savefig(project_metadata.plot_dir / f'{'ST Data'}.{file_ext}', 
                    transparent = transparent)

    if not display:
        plt.close(ax.figure)
