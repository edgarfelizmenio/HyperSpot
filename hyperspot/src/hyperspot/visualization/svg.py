import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from pathlib import Path

from ..metadata import ProjectMetadata
from ..utils import var_symbols, spatial_shape

def plot_region(adata, region_id, ax, region_key="domain"):
    sq.pl.spatial_scatter(
        adata,
        color=region_key,
        title=f"EC region: {region_id}",
    )

def plot_domain_svgs(adata, top_genes, regions = list[str],
                             figsize=(4, 4), n_cols = 4, use_raw=False,
                             save=True, transparent=True, file_ext='png',
                             generic_title=False, shape="auto"):

    project_metadata = ProjectMetadata()

    if len(regions) == 0 or len(top_genes) == 0:
        print("plot_domain_svgs: no domain ct-SVGs to plot (cell type too sparse / nothing passed "
              "threshold) — skipping.")
        return

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
            # .copy(): passing an AnnData *view* to sq.pl.spatial_scatter triggers the
            # ImplicitModificationWarning (it writes uns[color_key] onto the view).
            adata_region = adata[adata.obs['domain'] == str(region)].copy()
            adata_plot = adata.copy()

            # guard regions left empty by create_svg_list's winner-take-all merge
            # (every ct-SVG scored higher in another region) — idxmax on empty raises.
            sub = top_genes[top_genes['region'] == region]
            if len(sub) == 0:
                fig.delaxes(ax_region); fig.delaxes(ax_expr)
                continue
            gene_id = sub['scores'].idxmax()
            gene_name = var_symbols(adata).loc[gene_id]
            score = top_genes.loc[gene_id]['scores']
            
            if use_raw:
                vals = adata_plot.raw[:,gene_id].X
            else:
                vals = adata_plot[:,gene_id].X
            # remove zeros
            nz = vals[vals > 0]
            vmin = np.percentile(nz, 5)
            vmax = np.percentile(nz, 95)
            
            # spot marker: "auto" -> circle if H&E registered else plain points; override via shape=
            _shape = spatial_shape(adata) if shape == "auto" else shape
            sq.pl.spatial_scatter(
                adata_region,
                shape=_shape,
                color='domain',
                title=f"region {region}",
                ax=ax_region,
                legend_loc=None
            )
            expr_title = f"region {region} ct-SVG" if generic_title else f"{gene_name}: {score:0.4}"
            sq.pl.spatial_scatter(
                adata_plot,
                shape=_shape,
                color = gene_id,
                ax = ax_expr,
                title=expr_title,
                use_raw=use_raw,
                vmin=vmin,
                vmax=vmax,
                colorbar=False)
            ax_region.axis('off')
            ax_expr.axis('off')
        else:
            fig.delaxes(ax_region)
            fig.delaxes(ax_expr)

    plt.gcf().set_layout_engine("tight")
    
    if save:
        plt.savefig(project_metadata.plot_dir / f'top_de_genes.{file_ext}', transparent=True)
    
    plt.show()

            
def top_gene_regions(domain_svgs, rank_by: str = 'scores', per: str = 'top'):
    """Gene-centric inversion of the per-domain ct-SVG lists.

    Returns {gene_name: {'regions': [sorted region labels], 'score': max rank_by}}.
    per='top' : a gene's regions = domains where it is the rank-1 (highest rank_by) ct-SVG.
    per='all' : a gene's regions = every domain where it is a ct-SVG (broader spanning).
    Unlike create_svg_list (winner-take-all → one region per gene), a gene here keeps ALL
    the regions it tops, so genes spanning >1 region are preserved (STAMarker-style).
    """
    out = {}
    for region, df in domain_svgs.items():
        if df is None or len(df) == 0:
            continue
        if per == 'top':
            rows = [df.sort_values(rank_by, ascending=False).iloc[0]]
        else:
            rows = [df.iloc[i] for i in range(len(df))]
        for row in rows:
            g = str(row['names']); s = float(row[rank_by])
            e = out.setdefault(g, {'regions': [], 'score': s})
            e['regions'].append(str(region)); e['score'] = max(e['score'], s)
    for g in out:
        out[g]['regions'] = sorted(set(out[g]['regions']), key=lambda r: (len(r), r))
    return out


def plot_svgs_by_gene(adata, domain_svgs, rank_by: str = 'scores', per: str = 'top',
                      limit: int = 12, per_row: int = None, figsize=(4.4, 4.0), use_raw=False, shape="auto",
                      img: bool = False, img_res_key: str = 'lowres', img_alpha: float = 1.0, domain_palette=None,
                      save=True, transparent=True, file_ext='png', save_path=None,
                      rasterize=True, raster_dpi=150):
    """Per top gene: [domains where it is a top gene] | [its expression].

    Gene-centric counterpart to plot_domain_svgs — a gene that is a top gene in >1 region
    highlights all of them (no region is emptied by a winner-take-all merge). Genes are
    ordered by number-of-regions spanned (desc), then score.

    ``img`` toggles the tissue-image background (default off). ``limit`` caps the total number of genes shown (None = all). ``per_row`` is how many genes
    to place per row (columns); genes wrap into stacked blocks of two physical rows each
    (domain row + expression row). ``per_row=None`` keeps every gene on a single row.
    """
    project_metadata = ProjectMetadata()
    gr = top_gene_regions(domain_svgs, rank_by=rank_by, per=per)
    if not gr:
        print("plot_svgs_by_gene: no ct-SVGs to plot — skipping.")
        return
    genes = sorted(gr, key=lambda g: (len(gr[g]['regions']), gr[g]['score']), reverse=True)[:limit]
    syms = var_symbols(adata)
    _shape = spatial_shape(adata) if shape == "auto" else shape
    # stable per-domain colour (tab20) so each domain keeps the same colour across panels
    import matplotlib as _mpl
    _all_doms = sorted(adata.obs['domain'].astype(str).unique(),
                       key=lambda d: int(d) if str(d).isdigit() else d)
    if domain_palette is not None:
        _pal = list(domain_palette)
        _dom_color = {d: _pal[i % len(_pal)] for i, d in enumerate(_all_doms)}
    else:
        _cmap = plt.get_cmap('tab20')
        _dom_color = {d: _mpl.colors.to_hex(_cmap(i % 20)) for i, d in enumerate(_all_doms)}

    # paired rows (domain row + expression row) per gene; genes wrap at `per_row` columns into stacked blocks
    n = len(genes)
    width, height = figsize
    ncols = n if (per_row is None or per_row <= 0) else min(per_row, n)
    n_blocks = int(np.ceil(n / ncols))
    nrows = 2 * n_blocks
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width * ncols, height * nrows))
    axes = np.asarray(axes).reshape(nrows, ncols)
    for i, gene_id in enumerate(genes):
        regset = set(gr[gene_id]['regions']); score = gr[gene_id]['score']
        gene_name = syms.loc[gene_id] if gene_id in syms.index else gene_id
        blk, col = divmod(i, ncols)
        ax_region, ax_expr = axes[blk * 2][col], axes[blk * 2 + 1][col]

        # (row 0) ONLY the cells assigned to this gene's domain(s), each domain its own colour
        _rs = sorted(regset, key=lambda d: int(d) if str(d).isdigit() else d)
        a_reg = adata[adata.obs['domain'].astype(str).isin(regset)].copy()
        a_reg.obs['domain'] = pd.Categorical(a_reg.obs['domain'].astype(str), categories=_rs)
        a_reg.uns['domain_colors'] = [_dom_color[d] for d in _rs]   # scanpy/squidpy colour source
        sq.pl.spatial_scatter(
            a_reg, 
            shape=_shape, 
            color='domain', 
            ax=ax_region, 
            img=img, 
            img_res_key=img_res_key, 
            img_alpha=img_alpha,
            palette=_mpl.colors.ListedColormap([_dom_color[d] for d in _rs]),
            legend_loc=None,
            title=f"{gene_name}: region{'' if len(_rs)==1 else 's'} {', '.join(_rs)}")
        _lg = ax_region.get_legend()
        if _lg is not None:
            _lg.remove()

        # (right) expression
        vals = adata[:, gene_id].X
        vals = np.asarray(vals.todense()).ravel() if hasattr(vals, 'todense') else np.asarray(vals).ravel()
        nz = vals[vals > 0]
        vmin = float(np.percentile(nz, 5)) if nz.size else None
        vmax = float(np.percentile(nz, 95)) if nz.size else None
        sq.pl.spatial_scatter(adata.copy(), shape=_shape, color=gene_id, ax=ax_expr, img=img, img_res_key=img_res_key, img_alpha=img_alpha,
                              title=f"{gene_name}: {score:0.4}", use_raw=use_raw,
                              vmin=vmin, vmax=vmax, colorbar=False)
        if rasterize:   # rasterize the dense point + tissue-image layers; text/axes stay vector -> small PDF
            for _ax in (ax_region, ax_expr):
                for _art in list(_ax.collections) + list(_ax.images):
                    _art.set_rasterized(True)
        ax_region.axis('off'); ax_expr.axis('off')

    # blank any trailing cells in the last block (when n is not a multiple of ncols)
    for j in range(n, n_blocks * ncols):
        blk, col = divmod(j, ncols)
        axes[blk * 2][col].axis('off'); axes[blk * 2 + 1][col].axis('off')

    plt.gcf().set_layout_engine("tight")
    if save:
        plt.savefig(project_metadata.plot_dir / f'svgs_by_gene.{file_ext}', transparent=transparent)
    if save_path is not None:
        for _p in ([save_path] if isinstance(save_path, (str, Path)) else save_path):
            plt.savefig(_p, bbox_inches='tight', dpi=raster_dpi)
    plt.show()


def plot_domain_de_genes(adata,
                         region,
                         de_scores,
                         names,
                         n_cols = 4,
                         var_names_column = 'SYMBOL',
                         figsize: tuple = (20, 15), limit=4):
    
    assert limit % n_cols == 0
    adata_region = adata[adata.obs['domain'] == str(region)]

    fig, axes = plt.subplots(1, 2, figsize=(20,10))
    # display region
    sq.pl.spatial_scatter(
        adata_region,
        color='domain',
        title=f"region: {region}",
        ax=axes[0],
        legend_loc=None)

    if limit is None or limit > len(names):
        limit = len(names)

    adata_plot = adata.copy()
    # Dataset-agnostic: rename to the display column only if it exists (kidney SYMBOL sidecar);
    # for symbol-keyed data (no such column) the var_names already ARE the display names.
    if var_names_column is not None and var_names_column in adata.var.columns:
        adata_plot.var_names = adata.var[var_names_column].astype(str)
    adata_plot.var_names_make_unique()
    de_gene_idx = adata.var_names.get_indexer(names[:limit])
    genes = adata_plot.var_names[de_gene_idx]
    
    sc.pl.violin(
        adata_plot,
        keys = genes,
        ax = axes[1],
    )

    plt.gcf().set_layout_engine("tight")
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

    plt.gcf().set_layout_engine("tight")
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

    if len(ct_svgs) == 0:
        print(f"region {region}: no ct-SVGs to plot — skipping.")
        return

    adata_region = adata[adata.obs['domain'] == str(region)]

    fig, axes = plt.subplots(1, 2, figsize=(20,10))
    # display region
    sq.pl.spatial_scatter(
        adata_region,
        color='domain',
        title=f"region: {region}",
        ax=axes[0],
        legend_loc=None)

    if limit is None or limit > len(ct_svgs):
        limit = len(ct_svgs)

    adata_plot = adata.copy()
    ct_svgs = ct_svgs[:limit]
    gene_names = var_symbols(adata).loc[ct_svgs].tolist()
    
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
    
    plt.gcf().set_layout_engine("tight")
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

    plt.gcf().set_layout_engine("tight")
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

    plt.gcf().set_layout_engine("tight")
    plt.show()

def plot_ct_marker_groups_heatmap(adata, 
                           top_markers: dict, 
                           groupby="domain",
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