"""Old-vs-new HyperSpot check: does a code change keep the results?

Runs the same seeded steps with the code at a git ref (default HEAD, checked out in a temporary
worktree) and with the working tree, each in its own process on its own sandbox copy of a project,
then compares them. The production project is only read: the sandbox holds a copy of
project_metadata.pkl, a read-only symlink to hypergraph.pth and a copy of the trained model.

Steps (per arm):
  batches  first 3 training batches (seeded): cells, hyperedges, counts, incidence entries
  forward  model outputs on the first batch, same initial weights (seed 7), eval mode
  embed    node / hyperedge / neighborhood embeddings from the project's trained model
  train    --epochs epochs from scratch (seed 42): saved epoch, loss, weights, optimizer state

Comparisons:
  old vs new   batches must be identical; the rest identical with --bitwise, else within --rtol/--atol
  new vs new   the working tree run twice with the same seed: everything must be identical

Usage (from the repo root, GPU visible):
  python scripts/verify/compare_versions.py --project-dir "demo/hyperspot_project_files/Human Kidney Cancer Interface"
  options: --ref HEAD  --epochs 3  --sample 0.7  --batch-size 8192  --bitwise  --rtol 1e-4 --atol 1e-5  --keep
"""
import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

MODEL = "CTHGCAEv6Embedder"


# ---------------------------------------------------------------- one arm (runs in its own process)
def run_arm(a):
    os.chdir(a.root)
    import hyperspot
    from hyperspot.metadata import ProjectMetadata
    ProjectMetadata.use(a.rel)
    from hyperspot.data import STHypergraphData
    from hyperspot.utils import set_seed
    from hyperspot.config import AcceleratorOptions
    from hyperspot.pipeline import HypergraphEmbedderDriver, HypergraphEmbedderArgs
    from hyperspot.dataset.hypergraph import STHypergraphMatrixDataset

    cuda = AcceleratorOptions.CUDA
    folder = ProjectMetadata().project_folder
    hg = STHypergraphData.load()
    n_cells = hg["cell_feats"].shape[0]

    def model_args(hyperedge_head=True):
        # like the CLI: the hyperedge head is built for training (fit sets it), not for make-embedding
        args = HypergraphEmbedderArgs(model_name=MODEL, embedding_dim=a.embedding_dim, dropout=0.2,
                                      input_dim=hg["cell_feats"].shape[1])
        if hyperedge_head:
            args.hyperedge_feat_dim = hg["hyperedge_feats"].shape[1]
        return args

    def entries(H):
        # (row, col) of the nonzero incidence entries, sorted row-major
        return H.coalesce().indices().t().cpu() if H.is_sparse else H.nonzero().cpu()

    out = {"hyperspot_file": hyperspot.__file__}

    # batches
    set_seed(42)
    ds = STHypergraphMatrixDataset(hg, K=a.sample, M=a.sample, device="cuda")
    order = torch.randperm(n_cells).tolist()
    batches = []
    for b in range(3):
        idx = order[b * a.batch_size:(b + 1) * a.batch_size]
        if not idx:
            break
        batch = ds[idx]
        batches.append({
            "counts": torch.tensor([batch.sample_n_cells, batch.sample_n_hyperedges,
                                    batch.neighborhood_n_cells, batch.neighborhood_n_hyperedges]),
            "cells": batch.batch_subgraph_cells.cpu(),
            "hyperedges": batch.batch_subgraph_hyperedges.cpu(),
            "entries": entries(batch.batch_incidence_matrix),
        })
        if b == 0:
            first = batch
    out["batches"] = batches

    # forward
    set_seed(7)
    drv = HypergraphEmbedderDriver()
    drv._HypergraphEmbedderDriver__setup_model(model_args(), cuda)
    drv.model.eval()
    with torch.no_grad():
        res = drv.model(first.to_device(drv.device))
    out["forward"] = [r.cpu() for r in res if torch.is_tensor(r)]
    del ds, first, drv
    torch.cuda.empty_cache()

    # embed (the project's trained model, copied into the sandbox)
    set_seed(42)
    HypergraphEmbedderDriver().generate_embedding(hg, model_args(hyperedge_head=False), a.batch_size, cuda,
                                                  K=a.sample, M=a.sample)
    out["embed"] = {k: torch.load(folder / f"{k}_embeddings.pth", map_location="cpu")
                    for k in ("node", "hyperedge", "neighborhood")}

    # train (overwrites the sandbox model)
    set_seed(42)
    HypergraphEmbedderDriver().fit(hg, model_args(), a.epochs, a.batch_size, cuda, K=a.sample, M=a.sample)
    ck = torch.load(folder / f"{MODEL}.pth", map_location="cpu", weights_only=False)
    out["train"] = {"epoch": torch.tensor(ck["epoch"]), "loss": torch.tensor(ck["loss"]),
                    "weights": ck["embedder_state_dict"],
                    "optimizer": ck["optimizer_state_dict"]["state"]}
    torch.save(out, a.out)


# ---------------------------------------------------------------- orchestration and comparison
def flatten(o, prefix=""):
    if torch.is_tensor(o):
        yield prefix, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from flatten(v, f"{prefix}[{i}]")


def compare(x, y, exact, rtol, atol):
    """Rows of (name, ok, max |diff|)."""
    fx, fy = dict(flatten(x)), dict(flatten(y))
    rows = []
    for k in sorted(fx.keys() | fy.keys()):
        if k not in fx or k not in fy:
            rows.append((k, False, "missing in one arm"))
            continue
        a, b = fx[k], fy[k]
        if a.shape != b.shape:
            rows.append((k, False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"))
            continue
        if not a.is_floating_point():
            rows.append((k, torch.equal(a, b), "" if torch.equal(a, b) else "differs"))
            continue
        same_nan = torch.equal(a.isnan(), b.isnan())
        a0, b0 = a.nan_to_num(0.0).double(), b.nan_to_num(0.0).double()
        diff = (a0 - b0).abs().max().item() if a.numel() else 0.0
        ok = same_nan and (torch.equal(a0, b0) if exact else torch.allclose(a0, b0, rtol=rtol, atol=atol))
        rows.append((k, ok, f"{diff:.2e}" + ("" if same_nan else " (NaN pattern differs)")))
    return rows


def report(title, rows, show=12):
    bad = [r for r in rows if not r[1]]
    print(f"\n{title}: {len(rows) - len(bad)}/{len(rows)} tensors pass")
    for name, ok, info in (bad or rows)[:show]:
        print(f"   {'ok ' if ok else 'BAD'} {name:60s} {info}")
    if len(bad or rows) > show:
        print(f"   ... {len(bad or rows) - show} more")
    return not bad


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-dir", required=True, help="production project folder (only read)")
    p.add_argument("--ref", default="HEAD", help="git ref of the old code")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--sample", type=float, default=0.7, help="sampling fraction for hyperedges and cells")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--embedding-dim", type=int, default=20)
    p.add_argument("--bitwise", action="store_true", help="old vs new must be identical, not just close")
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--workdir", default=None, help="default: a new temporary folder")
    p.add_argument("--keep", action="store_true", help="keep the work folder and the old-code worktree")
    # internal: one arm
    p.add_argument("--arm", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--root", help=argparse.SUPPRESS)
    p.add_argument("--rel", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.arm:
        return run_arm(a)

    repo = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=Path(__file__).parent,
                               capture_output=True, text=True, check=True).stdout.strip())
    src = Path(a.project_dir).resolve()
    rel = str(pickle.load(open(src / "project_metadata.pkl", "rb"))["project_folder"])
    work = Path(a.workdir or tempfile.mkdtemp(prefix="hyperspot_verify_")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    old_src = work / "old_src"
    subprocess.run(["git", "worktree", "add", "--detach", str(old_src), a.ref], cwd=repo, check=True,
                   capture_output=True)
    print(f"old = {a.ref} ({old_src}), new = working tree ({repo}); work folder {work}")

    arms = {"old": old_src, "new": repo, "new_repeat": repo}
    outs = {}
    try:
        for name, code in arms.items():
            root = work / name
            d = root / rel
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / "project_metadata.pkl", d)
            shutil.copy(src / f"{MODEL}.pth", d)
            if not (d / "hypergraph.pth").exists():
                os.symlink(src / "hypergraph.pth", d / "hypergraph.pth")
            env = dict(os.environ, PYTHONPATH=str(code / "hyperspot" / "src"))
            cmd = [sys.executable, str(Path(__file__).resolve()), "--arm", "--root", str(root), "--rel", rel,
                   "--out", str(root / "out.pt"), "--project-dir", str(src), "--epochs", str(a.epochs),
                   "--sample", str(a.sample), "--batch-size", str(a.batch_size),
                   "--embedding-dim", str(a.embedding_dim)]
            print(f"running {name} ...", flush=True)
            subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
            outs[name] = torch.load(root / "out.pt", weights_only=False)
            want = code / "hyperspot" / "src"
            assert Path(outs[name]["hyperspot_file"]).resolve().is_relative_to(want.resolve()), \
                f"{name} imported {outs[name]['hyperspot_file']}, expected code from {want}"
    finally:
        if not a.keep:
            subprocess.run(["git", "worktree", "remove", "--force", str(old_src)], cwd=repo, capture_output=True)

    old, new, rep = outs["old"], outs["new"], outs["new_repeat"]
    ok = True
    ok &= report("old vs new: batches (must be identical)", compare(old["batches"], new["batches"], True, 0, 0))
    for step in ("forward", "embed", "train"):
        ok &= report(f"old vs new: {step} ({'bitwise' if a.bitwise else f'rtol {a.rtol}, atol {a.atol}'})",
                     compare(old[step], new[step], a.bitwise, a.rtol, a.atol))
    for step in ("batches", "forward", "embed", "train"):
        ok &= report(f"new vs new (same seed): {step} (must be identical)",
                     compare(new[step], rep[step], True, 0, 0))
    print("\nRESULT:", "PASS" if ok else "FAIL")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
