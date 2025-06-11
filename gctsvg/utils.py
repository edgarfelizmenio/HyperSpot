def get_gene_minmax(adata, gene_name):
    index = adata.var_names.get_loc(gene_name)
    gene_values = adata.X[:,index]
    gene_min = gene_values.min()
    gene_max = gene_values.max()

    return gene_min, gene_max
    