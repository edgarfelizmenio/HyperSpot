import ast
import json
import torch

import numpy as np
from skimage.draw import polygon
from pathlib import Path
import pandas as pd
import scanpy as sc

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

def filter(adata, is_mt_gene=None, min_genes=20, min_locs=20, max_genes=None, max_locs=None):
    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)

    # remove mitochondrial genes    
    if is_mt_gene:
        non_mt_genes = adata.var[~is_mt_gene(adata)].index.values
        adata = adata[:, non_mt_genes]
    
    # Use scanpy's filter_cells and filter_genes
    sc.pp.filter_cells(adata, min_genes=min_genes, max_genes=max_genes)
    sc.pp.filter_genes(adata, min_cells=min_locs, max_cells=max_locs)
    
    return adata

def compute_hvgs(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)

def preprocess(adata, target_sum = None, max_value = None, scale=False):
    adata.layers['original'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, zero_center=False, max_value=max_value)
        adata.X = adata.X.astype('float32')

def get_annotation(host="www.ensembl.org", file_dir="."):
    cache_path = Path(file_dir) / "GRCh38-2020-A_annotations.json"

    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if host in cache:
            print(f"Loaded annotation for '{host}' from cache.")
            return pd.DataFrame.from_dict(cache[host], orient='index')
    else:
        cache = {}
        
    # fetch from BioMart
    annot = sc.queries.biomart_annotations(
        "hsapiens",
        ["ensembl_gene_id", 
         "external_gene_name",
         "gene_biotype",
         "chromosome_name"],     
        host=host
    )
    
    protein_coding_genes = annot[annot['gene_biotype'] == 'protein_coding'].set_index('ensembl_gene_id')

    cache[host] = protein_coding_genes.to_dict(orient='index')
    cache_path.write_text(json.dumps(cache))
    print(f"Saved annotation for '{host}' to cache.")

    return protein_coding_genes

def find_conflicting_genes(adata_sc, adata_st, geneset: pd.DataFrame):
    st_map = adata_st.var[['SYMBOL']].copy()
    st_map = st_map[st_map.index.isin(geneset.index)]
    st_map = st_map[st_map['SYMBOL'].notna()]
    
    dupes = st_map[st_map['SYMBOL'].duplicated(keep=False)].copy()
    
    if len(dupes) == 0:
        print("No duplicate symbols found.")
        # return empty dataframe
        return pd.DataFrame()

    print(dupes)
    
    counts = np.array(adata_st[:, dupes.index].X.sum(axis=0)).flatten()
    dupes['total_counts'] = counts
    # percentage of spots that have nonzero expression of the gene
    dupes['pct_nonzero'] = (
        np.array((adata_st[:, dupes.index].X > 0).sum(axis=0)).flatten() / adata_st.n_obs
    )
    
    dupes['would_keep'] = False
    
    dupes.loc[dupes.groupby('SYMBOL', observed=True)['total_counts'].idxmax(), 'would_keep'] = True
    
    dupes['in_sc'] = dupes['SYMBOL'].isin(adata_sc.var_names)
    
    n_conflicts = dupes['SYMBOL'].nunique()
    n_relevant = dupes[dupes['in_sc']]['SYMBOL'].nunique()
    
    print(f"Found {n_conflicts} symbols with duplicate Ensembl IDs")
    print(f"  → {n_relevant} overlap with scRNA-seq and require manual inspection")
    print(f"  → {n_conflicts - n_relevant} are absent from scRNA-seq and will be handled automatically\n")
    
    return dupes.sort_values(['in_sc', 'SYMBOL', 'total_counts'], ascending=[False, True, False])    

def use_ensembl_gene_ids(
    adata_sc, 
    adata_st, 
    geneset: set,                     
    drop_cols: list[str] = [], 
    rename_cols: dict[str, str] = {}
):
    st_map = adata_st.var[['SYMBOL']].copy()
    st_map = st_map[st_map.index.isin(geneset.index)]
    st_map = st_map[st_map['SYMBOL'].notna()]
    
    st_map = st_map[~st_map.index.isin(drop_cols)]
    # st_map['SYMBOL'] = st_map.index.map(rename_cols).fillna(st_map['SYMBOL'].values)
    mask = st_map.index.isin(rename_cols)
    new_symbols = st_map.loc[mask].index.map(rename_cols)
    # symbol is categorical, make new categories first before assignment
    st_map['SYMBOL'] = st_map['SYMBOL'].cat.add_categories(
        [s for s in new_symbols if s not in st_map['SYMBOL'].cat.categories]
    )
    st_map.loc[mask, 'SYMBOL'] = new_symbols
    
    dupes = st_map[st_map['SYMBOL'].duplicated(keep=False)]
    if len(dupes):
        raise ValueError(
            f"Unresolved duplicate symbols remain after applying drop_cols/rename_cols. "
            f"Run check_symbol_conflicts() and update drop_cols/rename_cols accordingly.\n"
            f"{dupes.sort_values('SYMBOL')}"
        )
    
    common = st_map[st_map['SYMBOL'].isin(adata_sc.var_names)]
    
    new_adata_st = adata_st[:, common.index].copy()
    new_adata_sc = adata_sc[:, common['SYMBOL'].astype(str).tolist()].copy()
    new_adata_sc.var.index = common.index
    new_adata_sc.var_names_make_unique()
    
    n_genes = len(common)
    print(f"Retained {n_genes} genes common to both datasets")
    print(f"  scRNA-seq: {adata_sc.n_vars} → {n_genes} genes")
    print(f"  Spatial:   {adata_st.n_vars} → {n_genes} genes")
    
    return new_adata_sc, new_adata_st
    
# def use_ensembl_gene_ids(adata_sc, adata_st, host="www.ensembl.org", drop_cols: list[str]=[], rename_cols: dict[str, str] = {}):
#     annot = sc.queries.biomart_annotations("hsapiens",
#                                        ["ensembl_gene_id", 
#                                         "external_gene_name",
#                                         "gene_biotype",
#                                         "chromosome_name"],
#                                            host=host)
    
#     # retain genes of interest and genes with external gene names
#     annot = annot[(annot['gene_biotype'] == 'protein_coding') &
#                 annot['external_gene_name'].notna() & 
#                 (annot['chromosome_name'] != 'MT')]

#     # merge with genes of st data and remove/rename genes with same names if they have zero/negligible expression
#     annot_st = pd.merge(annot, adata_st.var, left_on='ensembl_gene_id', right_index=True, how='inner')
#     # drop columns
#     annot_st = annot_st[~annot_st['ensembl_gene_id'].isin(drop_cols)]
#     # rename columns
#     annot_st['SYMBOL'] = annot_st["ensembl_gene_id"].map(rename_cols).fillna(annot_st['SYMBOL'])
#     annot_st['external_gene_name'] = annot_st["ensembl_gene_id"].map(rename_cols).fillna(annot_st['external_gene_name'])

#     annot_st = annot_st.reset_index().set_index('ensembl_gene_id', drop=False)
#     annot_st = annot_st.rename_axis('ENSEMBL')
#     new_st_var = annot_st[adata_st.var.columns]
    
#     new_sc_var = pd.merge(annot_st, adata_sc.var, right_index=True, left_on='SYMBOL', how='inner')
#     new_sc_var = new_sc_var[['SYMBOL']]

#     new_adata_sc = adata_sc[:, new_sc_var['SYMBOL']].copy()
#     new_adata_sc.var = new_sc_var
#     new_adata_sc.var_names_make_unique()

#     new_adata_st = adata_st[:, new_st_var.index].copy()
#     new_adata_st.var = new_st_var
#     new_adata_st.var_names_make_unique()
    
#     return new_adata_sc, new_adata_st

def add_generated_cell_locations(spotiphy_proj_dir, adata_decomp, adata_st):

    # Load *n_cell_df.csv*. This file contains nucleus indices, nucleus centers, and cell counts per spot.
    # Note: Based on the information provided by *n_cell_df.csv*, Spotiphy predicts the number of cells of each type in a spot. 
    # However, identifying which of the cells are of the given types is a random guess.
    # Cell df contains nucleus indices, nucleus centers, and cell counts per spots
    cell_df = pd.read_csv(spotiphy_proj_dir / "segmentation" / "n_cell_df.csv",
                         index_col=0)
    
    # Load segmentation results.
    # Segmentation results contain cells' membership to spots, tissue location, and cell boundaries, which are important in visualization.
    segmentation_boundary = np.load(spotiphy_proj_dir / "segmentation" / "segmentation_boundary.npy", allow_pickle=True)
    # segmentation_boundary.shape
    
    # Extract nucleus centers and cell boundaries from the segmentation results.
    cell_centers = np.empty((adata_decomp.n_obs, 2), dtype=float)
    cell_boundaries = np.empty((adata_decomp.n_obs,32,2), dtype=float)
    
    for i in range(adata_st.n_obs):
        cell_count = cell_df['cell_count'][i]
        if cell_count == 0:
            continue
        nucleus_indices = ast.literal_eval(cell_df['Nucleus indices'][i])
        nucleus_centers = ast.literal_eval(cell_df['Nucleus centers'][i])
        nucleus_centers = np.array(nucleus_centers)
        cell_boundary_x = segmentation_boundary[nucleus_indices][:,:,0]
        cell_boundary_y = segmentation_boundary[nucleus_indices][:,:,1]
        
        # nucleus_centers_df = pd.DataFrame(nucleus_centers, columns=["nucleus_x", "nucleus_y"])
    
        cell_indexes = adata_decomp[adata_decomp.obs["spot_name"] == adata_st.obs_names[i]].obs_names
        cell_centers[cell_indexes.values.astype(int)] = nucleus_centers
        cell_boundaries[cell_indexes.values.astype(int)] = segmentation_boundary[nucleus_indices]
        # print(cell_boundary_x)
        # cell_topology.loc[cell_indexes, "boundary_x"] = cell_boundary_x
        # cell_topology.loc[cell_indexes, "boundary_y"] = cell_boundary_y

    adata_decomp.obsm['cell_nucleus'] = cell_centers

    set_segmentation_masks(adata_decomp, cell_boundaries)
    # adata_decomp.obsm['cell_nucleus'].shape, adata_decomp.obsm['cell_boundaries'].shape
    
    # Assign the nucleus location as the location of each cell. 
    # Squidpy's plotting functions require the input data to have the spatial metadata ` adata_decomp.obsm['spatial'] `.
    return adata_decomp

# TODO make crop coordinates match hires image data
def set_segmentation_masks(adata, segmentation_boundary):
    """
    Prepare segmentation for hires image only for Squidpy plotting.
    """
    library_id = list(adata.uns["spatial"].keys())[0]

    adata.obs['library_id'] = library_id
    
    # Hires image
    img = adata.uns["spatial"][library_id]["images"]["hires"]
    H, W = img.shape[:2]

    # Nuclei coordinates = hires
    nuclei = adata.obsm['cell_nucleus'].copy()
    adata.obsm['spatial'] = nuclei

    scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
    segmentation_boundary_scaled = segmentation_boundary * scale
    
    # Rasterize polygons into a segmentation mask
    seg_mask = np.zeros((H, W), dtype=np.int32)
    for i, poly in enumerate(segmentation_boundary_scaled, start=1):
        poly_int = np.round(poly).astype(np.int32)
        poly_int[:,0] = poly_int[:,0]
        poly_int[:,1] = poly_int[:,1]
        rr, cc = polygon(poly_int[:,1], poly_int[:,0], shape=seg_mask.shape)
        seg_mask[rr, cc] = i

    # Save the mask
    adata.uns["spatial"][library_id]["images"]["segmentation"] = seg_mask
    n_cells = len(adata.obs)
    n_seg   = int(seg_mask.max())
    seg_ids = np.zeros(n_cells, dtype=np.int32)
    seg_ids[:min(n_seg, n_cells)] = np.arange(1, min(n_seg, n_cells) + 1)
    adata.obs["segmentation"] = seg_ids
    
    print(f"Prepared hires segmentation: {seg_mask.shape}, {seg_mask.max()} cells")


def get_common_geneset(adata_st, adata_sc):
    st_hvg = adata_st.var[adata_st.var['highly_variable']].index
    sc_hvg = adata_sc.var[adata_sc.var['highly_variable']].index

    geneset = st_hvg.intersection(sc_hvg)

    print(f"Number of overlap genes: {len(geneset)}")
    return geneset

def get_feature_matrix(adata):
    if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
       feat = adata.X.toarray()
    else:
       feat = adata.X
    print(type(feat))
    return torch.tensor(feat)