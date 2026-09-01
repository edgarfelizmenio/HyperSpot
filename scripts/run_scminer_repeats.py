"""
run_scminer_repeats.py — Run MICA clustering N times (unseeded) per cell type to
estimate scMiner's run-to-run variance, for a fair distribution-vs-distribution
comparison against HyperSpot's 10-seed domains.

MICA (`mica ge`) exposes NO seed flag; its node2vec walks / HNSW / Louvain stages
are stochastic, so repeated invocations naturally vary. We therefore approximate a
"seed distribution" by simply repeating the run.

Must be run in the scMiner conda environment. CPU-only (no GPU).

    conda run -n scMiner python scripts/run_scminer_repeats.py \
        --iscRNA   <path/to/ST_decomposition.h5ad> \
        --out_root <path/to/scminer_clusters_runs/> \
        --n_runs 10 [--cts RCC Myeloid T-cell] [--resolution 1.822] [--num_workers 2]

Output: <out_root>/run_<i>/<ct_safe>.csv  with columns: cell_id, cluster
Existing (complete) run/ct CSVs are skipped, so the job is resumable.
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


def run_mica(h5ad_path, out_dir, resolution, num_workers, n_cells,
             nnm=None, nne=None, annef=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if nnm is None:   nnm   = min(80, n_cells // 2)
    if nne is None:   nne   = min(20, n_cells - 1)
    if annef is None: annef = max(200, nnm * 2)
    cmd = ["mica", "ge", "-i", str(h5ad_path), "-o", str(out_dir),
           "-res", str(resolution), "-nw", str(num_workers),
           "-nnm", str(nnm), "-nne", str(nne), "-annef", str(annef)]
    print(f"    Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"MICA failed (nnm={nnm}, nne={nne}, annef={annef}, "
                           f"rc={result.returncode})")
    return out_dir


def cluster_ct(adata_ct, out_csv, resolution, num_workers, tmp):
    safe = out_csv.stem
    n = adata_ct.n_obs
    ct_h5ad = tmp / f"{safe}.h5ad"
    adata_ct.write_h5ad(str(ct_h5ad))
    mica_out = tmp / f"{safe}_mica"
    attempts = [
        {},
        {"nnm": min(30, max(5, n // 4)), "nne": min(15, n - 1), "annef": 500},
        {"nnm": min(15, max(5, n // 8)), "nne": min(10, n - 1), "annef": 1000},
    ]
    for k, params in enumerate(attempts):
        if mica_out.exists():
            shutil.rmtree(mica_out)
        try:
            run_mica(ct_h5ad, mica_out, resolution, num_workers, n_cells=n, **params)
            break
        except RuntimeError as e:
            if k == len(attempts) - 1:
                raise
            print(f"    attempt {k+1} failed; retry fewer neighbors. [{e}]", flush=True)

    txt_files = list(mica_out.glob("clustering_*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No clustering_*.txt in {mica_out}")
    result = pd.read_csv(txt_files[0], sep="\t", index_col=0)
    labels = pd.DataFrame({"cell_id": result.index, "cluster": result["label"].values})
    labels.to_csv(out_csv, index=False)
    print(f"    saved {len(labels)} labels -> {out_csv} "
          f"({labels['cluster'].nunique()} clusters)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iscRNA", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--cts", nargs="+", default=["RCC", "Myeloid", "T-cell"])
    ap.add_argument("--ct_key", default="cell_type")
    ap.add_argument("--resolution", type=float, default=1.822)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    print(f"Loading {args.iscRNA} ...", flush=True)
    adata = ad.read_h5ad(args.iscRNA)
    subsets = {}
    for ct in args.cts:
        sub = adata[adata.obs[args.ct_key] == ct].copy()
        subsets[ct] = sub
        print(f"  {ct}: {sub.n_obs} cells", flush=True)

    with tempfile.TemporaryDirectory(prefix="mica_rep_") as tmp:
        tmp = Path(tmp)
        for i in range(args.n_runs):
            run_dir = out_root / f"run_{i}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n===== run {i} =====", flush=True)
            for ct in args.cts:
                out_csv = run_dir / f"{ct_safe(ct)}.csv"
                if out_csv.exists():
                    print(f"  [{ct}] run {i} already done — skip", flush=True)
                    continue
                sub = subsets[ct]
                if sub.n_obs < 10:
                    print(f"  [{ct}] too few cells — skip", flush=True)
                    continue
                print(f"  [{ct}] run {i}", flush=True)
                cluster_ct(sub, out_csv, args.resolution, args.num_workers, tmp)
    print("\nAll runs done.", flush=True)


if __name__ == "__main__":
    main()
