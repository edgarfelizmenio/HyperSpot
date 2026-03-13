import scanpy as sc
import torch

from pathlib import Path

from .metadata import ProjectMetadata

def load_orig_st_data():
    adata_st = sc.read_h5ad(ProjectMetadata().project_folder / 'ST_init.h5ad')
    return adata_st

def load_decomposed_st_data():
    adata_decomp = sc.read_h5ad(ProjectMetadata().project_folder / 'ST_decomposition.h5ad')
    return adata_decomp
    
def save_ct_spatial_data(adata_ct, cell_type: str):
    # save adata for cell type
    adata_ct.write_h5ad(ProjectMetadata().ct_data_dir / f'ST_cell_{cell_type}.h5ad')

def load_ct_spatial_data(cell_type: str):
    # load adata for cell type
    adata_ct = sc.read_h5ad(ProjectMetadata().ct_data_dir / f"ST_cell_{cell_type}.h5ad")
    return adata_ct

def get_gene_minmax(adata, gene_name):
    index = adata.var_names.get_loc(gene_name)
    gene_values = adata.X[:,index]
    gene_min = gene_values.min()
    gene_max = gene_values.max()

    return gene_min, gene_max

def build_incidence_matrix(hyperedge_index):
    # reindex hyperedges to make compact incidence matrix for a batch
    unique_cell_idx, map_to_unique_cell = hyperedge_index[0].unique(return_inverse=True)
    unique_neighborhood_idx, map_to_unique_neighborhood = hyperedge_index[1].unique(return_inverse=True)

    n_unique_cells = unique_cell_idx.shape[0]
    n_unique_neighborhoods = unique_neighborhood_idx.shape[0]
    
    incidence_matrix = torch.zeros((n_unique_cells, n_unique_neighborhoods), dtype=torch.float)
    incidence_matrix[map_to_unique_cell, map_to_unique_neighborhood] = 1

    return incidence_matrix