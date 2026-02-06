import scanpy as sc
import squidpy as sq

def plot_region(adata, region_id, ax, region_key="leiden"):
    sq.pl.spatial_scatter(
        adata,
        color=region_key,
        title=f"EC region: {region_id}",
    )
    