"""
run_scminer_clustering.py — Run MICA clustering on Spotiphy iscRNA data per cell type.

Must be run in the scMiner conda environment:
    conda run -n scMiner python scripts/run_scminer_clustering.py \
        --iscRNA  <path/to/ST_decomposition.h5ad> \
        --out_dir <path/to/scminer_clusters/> \
        [--ct_key cell_type] \
        [--resolution 1.822] \
        [--num_workers 4]

Output: one CSV per cell type at <out_dir>/<ct_safe>.csv with columns:
    cell_id, cluster
These can be loaded by ClusterEvaluation.compute_CT_spotiphy_scores() in the
HyperSpot environment without requiring scMiner.
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd


def ct_safe(ct: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", ct)


def run_mica(h5ad_path: Path, out_dir: Path, resolution: float, num_workers: int,
             n_cells: int = 10000, nnm: int = None, nne: int = None,
             annef: int = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if nnm is None:   nnm   = min(80, n_cells // 2)
    if nne is None:   nne   = min(20, n_cells - 1)
    if annef is None: annef = max(200, nnm * 2)
    cmd = [
        "mica", "ge",
        "-i", str(h5ad_path),
        "-o", str(out_dir),
        "-res", str(resolution),
        "-nw", str(num_workers),
        "-nnm", str(nnm),
        "-nne", str(nne),
        "-annef", str(annef),
    ]
    print(f"  Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"MICA failed (nnm={nnm}, nne={nne}, annef={annef}, "
                           f"return code {result.returncode})")
    return out_dir  # caller reads clustering_*.txt from here


def main():
    parser = argparse.ArgumentParser(description="Run MICA clustering on Spotiphy iscRNA per CT")
    parser.add_argument("--iscRNA",      required=True,  help="Path to ST_decomposition.h5ad (Spotiphy iscRNA)")
    parser.add_argument("--out_dir",     required=True,  help="Output directory for per-CT cluster CSVs")
    parser.add_argument("--ct_key",      default="cell_type", help="obs column with cell type labels")
    parser.add_argument("--resolution",  default=1.822,  type=float, help="MICA Louvain resolution")
    parser.add_argument("--num_workers", default=4,      type=int,   help="MICA parallel workers")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading iscRNA data from {args.iscRNA} …", flush=True)
    adata = ad.read_h5ad(args.iscRNA)
    cell_types = sorted(adata.obs[args.ct_key].unique().tolist())
    print(f"Cell types ({len(cell_types)}): {cell_types}", flush=True)

    with tempfile.TemporaryDirectory(prefix="mica_tmp_") as tmp:
        tmp = Path(tmp)
        for ct in cell_types:
            safe = ct_safe(ct)
            out_csv  = out_dir / f"{safe}.csv"
            emb_out  = out_dir / f"{safe}_embedding.csv"
            if out_csv.exists() and emb_out.exists():
                print(f"  [{ct}] already complete — skipping", flush=True)
                continue

            print(f"\n[{ct}]", flush=True)
            adata_ct = adata[adata.obs[args.ct_key] == ct].copy()
            print(f"  {adata_ct.n_obs} cells × {adata_ct.n_vars} genes", flush=True)

            if adata_ct.n_obs < 10:
                print(f"  Too few cells — skipping", flush=True)
                continue

            # write CT subset to temp h5ad for MICA
            ct_h5ad = tmp / f"{safe}.h5ad"
            adata_ct.write_h5ad(str(ct_h5ad))

            mica_out = tmp / f"{safe}_mica"
            # MICA's HNSW ANN fails on small cell types ("ef or M is too small")
            # when nnm (neighbors) is a large fraction of the cell count. Retry with
            # progressively fewer neighbors and a larger search ef before giving up.
            _n = adata_ct.n_obs
            _attempts = [
                {},                                                                  # default heuristic
                {"nnm": min(30, max(5, _n // 4)), "nne": min(15, _n - 1), "annef": 500},
                {"nnm": min(15, max(5, _n // 8)), "nne": min(10, _n - 1), "annef": 1000},
            ]
            for _k, _params in enumerate(_attempts):
                if mica_out.exists():
                    shutil.rmtree(mica_out)
                try:
                    run_mica(ct_h5ad, mica_out, args.resolution, args.num_workers,
                             n_cells=_n, **_params)
                    break
                except RuntimeError as _e:
                    if _k == len(_attempts) - 1:
                        raise
                    print(f"  MICA attempt {_k + 1} failed; retrying with fewer "
                          f"neighbors / larger ef. [{_e}]", flush=True)

            # MICA writes clusters + embedding to clustering_*.txt (not to clustered.h5ad)
            txt_files = list(mica_out.glob("clustering_*.txt"))
            if not txt_files:
                raise FileNotFoundError(f"No clustering_*.txt found in {mica_out}")
            # pick the file corresponding to the target resolution
            txt_file = txt_files[0]
            result = pd.read_csv(txt_file, sep="\t", index_col=0)
            # result columns: X, Y, label

            labels = pd.DataFrame({
                "cell_id": result.index,
                "cluster": result["label"].values,
            })
            labels.to_csv(out_csv, index=False)
            print(f"  Saved {len(labels)} labels → {out_csv}  "
                  f"({labels['cluster'].nunique()} clusters)", flush=True)

            # save 20D node2vec embedding; fall back to 2D XY only if not found
            emb_out = out_dir / f"{safe}_embedding.csv"
            node2vec_files = list(mica_out.glob("knn_*_graph_emb_on_gene_to_*.txt"))
            if node2vec_files:
                # node2vec format: first line is "n_nodes n_dims", rest is space-separated
                # index is integer position (0..n-1) → map to original obs_names
                emb20 = pd.read_csv(node2vec_files[0], sep=' ', header=None,
                                    skiprows=1, index_col=0)
                emb20 = emb20.sort_index()   # ensure row order matches adata_ct
                emb20.index = adata_ct.obs_names  # map positions to cell barcodes
                emb20.columns = [f"dim_{i}" for i in range(emb20.shape[1])]
                emb20.index.name = "cell_id"
                emb20.to_csv(emb_out)
                print(f"  Saved {emb20.shape[1]}D node2vec embedding → {emb_out}", flush=True)
            else:
                emb_df = pd.DataFrame({
                    "cell_id": result.index,
                    "X": result["X"].values,
                    "Y": result["Y"].values,
                })
                emb_df.to_csv(emb_out, index=False)
                print(f"  node2vec not found — saved 2D XY fallback → {emb_out}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
