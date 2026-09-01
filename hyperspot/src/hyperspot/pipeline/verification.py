"""
Data-property verification for HyperSpot.

Standalone checks that validate the assumptions the pipeline relies on, so they are
explicitly confirmed rather than assumed (useful for reproducibility / methods).

Currently: spot-duplication -- Spotiphy-style deconvolution assigns one per-type
expression profile per spot, so same-spot same-CT cells are EXACT duplicates
(identical expression -> identical embedding). The dedup used in clustering and
ctSVG detection is lossless *only because* of this. This module measures it and
reports per cell type. It shares `within_spot_max_deviation` with the dedup guards
(utils.py), so what is verified is exactly what the dedup relies on.
"""
import numpy as np
import pandas as pd

from ..metadata import ProjectMetadata
from ..utils import within_spot_max_deviation, load_ct_spatial_data


def verify_spot_duplication(expr_tol=1e-3, emb_tol=1e-3, embedding_key="embedding",
                            sample=None, verbose=True):
    """Verify same-spot same-CT cells are exact duplicates, per cell type.

    For each CT: max within-spot deviation of expression (adata.X) and, if present,
    of the embedding (adata.obsm[embedding_key]); the number of multi-cell spots;
    and pass/fail (deviation < tol). Returns a DataFrame indexed by cell type.

    Parameters
    ----------
    expr_tol, emb_tol : float
        Deviation below which same-spot cells count as identical.
    sample : int or None
        Limit the expression check to this many multi-cell spots (wide matrices);
        None checks all cells. The embedding is narrow so it is always checked full.
    """
    import scipy.sparse as sp
    pm = ProjectMetadata()
    rows = []
    for ct in pm.ct_list:
        a = load_ct_spatial_data(ct)
        if 'spot_name' not in a.obs or a.n_obs == 0:
            continue
        spots = a.obs['spot_name'].to_numpy()
        _, inv = np.unique(spots, return_inverse=True)
        n_multi = int((np.bincount(inv) >= 2).sum())

        expr_dev = within_spot_max_deviation(a.X, spots, sample=sample)
        emb_dev = (within_spot_max_deviation(np.asarray(a.obsm[embedding_key]), spots)
                   if embedding_key in a.obsm else np.nan)

        rows.append({
            "cell_type": ct,
            "n_cells": a.n_obs,
            "n_unique_spots": len(np.unique(inv)),
            "n_multi_cell_spots": n_multi,
            "expr_max_dev": expr_dev,
            "emb_max_dev": emb_dev,
            "duplicates": bool(expr_dev < expr_tol),
        })

    df = pd.DataFrame(rows).set_index("cell_type")
    if verbose and len(df):
        n_dup = int(df["duplicates"].sum())
        print(df.to_string())
        print(f"\n{n_dup}/{len(df)} cell types have exact same-spot duplicates "
              f"(max expr deviation < {expr_tol}).")
        if n_dup == len(df):
            print("=> spot-level dedup is LOSSLESS for this dataset.")
        elif n_dup == 0:
            print("=> genuine per-cell variation: dedup will be a no-op; standard per-cell path used.")
        else:
            print("=> MIXED: some CTs have per-cell variation. The dedup guard is per-dataset "
                  "all-or-nothing, so it falls back to per-cell whenever the sampled check finds variation.")
    return df
