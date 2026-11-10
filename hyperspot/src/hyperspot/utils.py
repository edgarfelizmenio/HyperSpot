import random

import scanpy as sc
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pathlib import Path

from .metadata import ProjectMetadata


def set_seed(seed: int = 42):
    """Seed every RNG HyperSpot uses (python, numpy, torch CPU/CUDA) and make cuDNN deterministic.

    Seeding happens once at the entry point, not inside library functions: each CLI
    command seeds at program start, and run_eval seeds at the start of every run
    (train -> embed -> cluster -> score). In notebooks / the Python API, call this once
    before running the pipeline steps; results then depend only on the seed and the
    order of the steps. Leiden clustering takes its own `seed` argument separately.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- spot-duplicate primitives (single source of truth) ------------------------
# Spotiphy-style deconvolution assigns one per-type profile per spot, so same-spot
# same-CT cells are exact duplicates (identical expression -> identical embedding).
# The dedup used in clustering + ctSVG detection, and the standalone verification,
# all rest on the two functions below so the check can never silently drift.

def within_spot_max_deviation(X, spot_labels, sample=None, seed=0):
    """Max abs deviation of any cell from its spot's first cell, over features.

    ~0 => same-spot cells are (near-)identical duplicates. `sample` limits the check
    to that many randomly chosen multi-cell spots (for wide/sparse expression
    matrices where materialising all cells is costly); None checks every cell.
    """
    spot_labels = np.asarray(spot_labels)
    n = len(spot_labels)
    if n == 0:
        return 0.0
    _, inv = np.unique(spot_labels, return_inverse=True)
    if sample is None:
        Xd = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
        rep = pd.Series(np.arange(n)).groupby(inv).first().sort_index().to_numpy()
        return float(np.abs(Xd - Xd[rep][inv]).max())
    multi = np.where(np.bincount(inv) >= 2)[0]
    if len(multi) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    check = multi if len(multi) <= sample else rng.choice(multi, sample, replace=False)
    dev = 0.0
    for g in check:
        members = np.where(inv == g)[0]
        Xm = X[members]
        Xm = np.asarray(Xm.todense()) if sp.issparse(Xm) else np.asarray(Xm)
        dev = max(dev, float(np.abs(Xm - Xm[0]).max()))
    return dev


def within_spot_deviation_per_spot(X, spot_inv):
    """Max abs within-spot deviation from each spot's first cell, PER spot.

    Returns array `dev` indexed by spot id (the np.unique inverse ordering); dev[s] ~ 0
    means spot s's cells are (near-)identical. Only multi-cell spots can exceed 0.
    Unlike `within_spot_max_deviation` (a single global max), this keeps a per-spot
    value so the dedup guard can decide spot-by-spot instead of all-or-nothing.
    """
    n = len(spot_inv)
    n_spots = int(spot_inv.max()) + 1 if n else 0
    dev = np.zeros(n_spots)
    if n == 0:
        return dev
    if not sp.issparse(X):                               # dense (e.g. narrow embedding): vectorised
        Xd = np.asarray(X)
        rep = pd.Series(np.arange(n)).groupby(spot_inv).first().sort_index().to_numpy()
        d = np.abs(Xd - Xd[rep[spot_inv]]).max(axis=1)  # per-cell dev from its spot's rep
        np.maximum.at(dev, spot_inv, d)                 # reduce to per-spot max
        return dev
    order = np.argsort(spot_inv, kind='stable')          # sparse/wide: group-wise, low peak mem
    ss = spot_inv[order]
    uniq, starts, counts = np.unique(ss, return_index=True, return_counts=True)
    for u, st, c in zip(uniq, starts, counts):
        if c < 2:
            continue
        Xm = np.asarray(X[order[st:st + c]].todense())
        dev[u] = float(np.abs(Xm - Xm[0]).max())
    return dev


def guarded_spot_reps(X, spot_labels, dedup=True, tol=1e-3, sample=None):
    """(rep_idx, inv) collapsing same-spot cells to one representative -- PER SPOT:
    a spot is collapsed only when its own cells are (near-)identical (within-spot
    deviation < tol); spots with genuine within-spot variation keep all their cells.

    This is robust to a few anomalous spots: with the old global-max gate, a SINGLE
    outlier spot (e.g. one spot whose same-CT cells embed differently) disabled dedup
    for the entire cell type, letting Leiden shatter the remaining distance-0 duplicate
    cliques (observed on breast: Invasive_Tumor 256 domains, Macrophages_1 66, vs 9-14
    for the deduped cell types). Now that one spot's cells are simply kept un-collapsed.

    dedup=False / no spot labels -> identity mapping (every cell its own unit). rep_idx[r]
    is a representative cell of unit r; inv[c] is the unit id of cell c (labels[inv]
    broadcasts unit labels back to cells).
    """
    n = X.shape[0]                                      # one unit per cell (row of X)
    inv = np.arange(n)                                  # identity: no dedup
    if dedup and n > 0 and spot_labels is not None:
        _, cand = np.unique(np.asarray(spot_labels), return_inverse=True)  # cand[c] = spot id
        # cheap global pre-check (honours `sample`): common case is ALL spots identical
        # -> collapse every spot, same fast path as before.
        if within_spot_max_deviation(X, spot_labels, sample=sample) < tol:
            inv = cand
        else:
            # some spot genuinely varies: full per-spot check, collapse only the
            # (near-)identical spots; divergent spots' cells stay as individual units.
            dev = within_spot_deviation_per_spot(X, cand)
            mask = (dev < tol)[cand]                     # cells whose spot is collapsible
            inv = np.arange(n)
            inv[mask] = n + cand[mask]                   # shared per-spot unit ids (disjoint from singletons)
            _, inv = np.unique(inv, return_inverse=True)  # make unit ids contiguous
    rep_idx = pd.Series(np.arange(n)).groupby(inv).first().sort_index().to_numpy()
    return rep_idx, inv
# -------------------------------------------------------------------------------

def save_orig_st_data(adata_st):
    adata_st.write_h5ad(ProjectMetadata().project_folder / 'ST_orig.h5ad')

def var_symbols(adata):
    """Human-readable gene symbols as a Series indexed by ``adata.var_names``.

    This is the SINGLE point at which the generic pipeline/visualization code learns
    how to label genes, and the only place in generic code that knows about the
    legacy ``'SYMBOL'`` column. Behaviour:

    * if a ``var['SYMBOL']`` column exists -> use it (ENSG-keyed data, e.g. the kidney
      10x CellRanger schema, carries symbols in this sidecar column);
    * otherwise -> fall back to ``var_names`` (symbol-keyed data, e.g. the breast ST
      reference, needs no sidecar: the symbols ARE the var_names).

    Keeping every generic SYMBOL access routed through here makes the library
    dataset-agnostic (option A) and is the designated migration point to option B:
    once no dataset carries a ``'SYMBOL'`` column (kidney var_names renamed to symbols
    via preprocessing.kidneycancer), this function collapses to ``return
    pd.Series(adata.var_names.astype(str), index=adata.var_names)`` with no change to
    any call site.
    """
    if "SYMBOL" in adata.var.columns:
        return adata.var["SYMBOL"].astype(str)
    return pd.Series(np.asarray(adata.var_names, dtype=str), index=adata.var_names)


def spatial_shape(adata):
    """squidpy spot `shape` appropriate to the platform, dataset-agnostically.

    This is the DEFAULT resolver used when a plotting function is called with ``shape="auto"``; callers can
    always override by passing an explicit ``shape`` ("circle"/"square"/"hex"/None).

    Spots are circular capture features on every platform (Visium 55um, original ST 100um); the grid
    arrangement (hex vs square) drives neighbour counts, not the marker. So the physically accurate, uniform
    default is a circle wherever an H&E is registered, and a plain point scatter where none is:

    * ``uns['spatial']`` present (image registered) -> ``"circle"`` (overlays the H&E);
    * no ``uns['spatial']`` -> ``None`` (plain ``obsm['spatial']`` scatter, points only).
    """
    return "circle" if "spatial" in adata.uns else None


def load_orig_st_data():
    adata_st = sc.read_h5ad(ProjectMetadata().project_folder / 'ST_orig.h5ad')
    return adata_st

def save_decomposed_st_data(adata_decomp):
    adata_decomp.write_h5ad(ProjectMetadata().project_folder / 'ST_decomposition.h5ad')

def load_decomposed_st_data():
    adata_decomp = sc.read_h5ad(ProjectMetadata().project_folder / 'ST_decomposition.h5ad')
    return adata_decomp

def save_aggr_st_data(adata_aggr):
    adata_aggr.write_h5ad(ProjectMetadata().project_folder / 'ST_aggregated.h5ad')

def load_aggr_st_data():
    adata_aggr = sc.read_h5ad(ProjectMetadata().project_folder / 'ST_aggregated.h5ad')
    if 'domain' in adata_aggr.obs:   # ensure discrete for plotting (squidpy/scanpy)
        adata_aggr.obs['domain'] = adata_aggr.obs['domain'].astype('category')
    return adata_aggr

# The H&E image is stored once, in ST_orig.h5ad. Per-cell-type files are written
# without uns['spatial'] (it used to be duplicated into every cell type, ~1 GB each)
# and gzip-compressed (Spotiphy same-spot duplicates compress ~13x, values unchanged).
# load_ct_spatial_data re-attaches the image in memory, so plotting is unaffected.
_ST_SPATIAL_UNS = {}

def _st_spatial_uns():
    """uns['spatial'] of ST_orig.h5ad, read once per project and shared (not copied)."""
    pf = ProjectMetadata().project_folder
    if pf not in _ST_SPATIAL_UNS:
        _ST_SPATIAL_UNS[pf] = sc.read_h5ad(pf / 'ST_orig.h5ad', backed='r').uns.get('spatial')
    return _ST_SPATIAL_UNS[pf]

def save_ct_spatial_data(adata_ct, cell_type: str):
    # save adata for cell type, without the image (restored on the in-memory object)
    spatial = adata_ct.uns.pop('spatial', None)
    try:
        adata_ct.write_h5ad(ProjectMetadata().ct_data_dir / f'ST_cell_{cell_type}.h5ad',
                            compression='gzip')
    finally:
        if spatial is not None:
            adata_ct.uns['spatial'] = spatial

def load_ct_spatial_data(cell_type: str, with_image: bool = True):
    # load adata for cell type; legacy files that still carry their own image keep it
    adata_ct = sc.read_h5ad(ProjectMetadata().ct_data_dir / f"ST_cell_{cell_type}.h5ad")
    if 'domain' in adata_ct.obs:      # ensure discrete for plotting (squidpy/scanpy)
        adata_ct.obs['domain'] = adata_ct.obs['domain'].astype('category')
    if with_image and 'spatial' not in adata_ct.uns:
        spatial = _st_spatial_uns()
        if spatial is not None:
            adata_ct.uns['spatial'] = spatial    # shared reference: treat as read-only
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