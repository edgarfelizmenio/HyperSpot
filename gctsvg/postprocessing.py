import anndata as ad
import numpy as np
from scipy import sparse

def build_adata(adata_orig, new_X):
    # Match the format of adata.X
    if sparse.issparse(adata_orig.X):
        # If original X is sparse, convert new_X_np to the same sparse format
        new_X = type(adata_orig.X)(new_X)
    else:
        # Otherwise, it's a dense numpy array
        new_X = new_X.astype(adata_orig.X.dtype)  # Match dtype just in case

    # Create new AnnData with the new X and same metadata
    adata_mod = ad.AnnData(X = new_X, 
                           # obs = adata_orig.obs.copy(), 
                           # var = adata_orig.var.copy(), 
                           uns = adata_orig.uns.copy(),
                            # Optional: also copy obsm, varm, layers, etc., if needed
                            obsm = adata_orig.obsm.copy(),
                            varm = adata_orig.varm.copy(),
                            layers = adata_orig.layers.copy())
                            # obs_names = adata_orig.obs_names.copy(),
                            # var_names = adata_orig.var_names.copy())
    adata_mod.obs_names = adata_orig.obs_names.astype("str")
    adata_mod.var_names = adata_orig.var_names.astype("str")
    return adata_mod

def build_output_objects(adata_orig, decomposition, reconstruction, cell_types):
    adata_decomposition = {cell_types[i]: build_adata(adata_orig, decomposition[i])
                           for i in range(len(cell_types))}

    adata_reconstruction = build_adata(adata_orig, reconstruction)

    return adata_decomposition, adata_reconstruction