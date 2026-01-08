import torch

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