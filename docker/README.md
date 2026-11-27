# HyperSpot Docker image

Image: `edgarfelizmenio/hyperspot:1.0`. Conda env `HyperSpot` with only what the `hyperspot` package uses
(`HyperSpot_environment.yml`), pinned to the versions of the env that produced the current results: Python 3.12.0,
torch 2.6.0 + CUDA 12.4, numpy 2.3.5, scipy 1.16.3, scikit-learn 1.8.0, anndata 0.12.0, scanpy 1.11.5,
squidpy 1.7.0, igraph 1.0.0, umap-learn 0.5.11, leidenalg 0.11.0, …; plus the `hyperspot` package from this repo
as of the build. No PyTorch Geometric (the hypergraph kNN uses scipy). Analysis-only packages used by notebooks
(gseapy, lifelines, …) are not in the image.

The image holds the code as it was when it was built: after code changes, rebuild and push under a new tag.

GPU: the torch wheels bundle CUDA 12.4; the host only needs an NVIDIA driver that supports CUDA 12.4
(tested with 550).

## Build and push

Run from the HyperSpot repo root (the build copies `hyperspot/` and `docker/HyperSpot_environment.yml`):

```bash
docker build -f docker/Dockerfile -t edgarfelizmenio/hyperspot:1.0 .
docker push edgarfelizmenio/hyperspot:1.0
```

Check the image (GPU visible, CLI works):

```bash
docker run --rm --gpus all edgarfelizmenio/hyperspot:1.0 \
  conda run --no-capture-output -n HyperSpot python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
docker run --rm edgarfelizmenio/hyperspot:1.0 conda run --no-capture-output -n HyperSpot hyperspot --help
```

## Run on the k8s server

HyperSpot reads and writes projects under `hyperspot_project_files/` in the current directory
(`settings.ini`: `ProjectFilesDir`), so `cd` to the folder that holds `hyperspot_project_files/` first.
Request one GPU (`-g 1`); one training per GPU.

```bash
python3 k8s_create_job.py -i edgarfelizmenio/hyperspot:1.0 -g 1 -n "mlb-edgar-<num>" \
  -c "cd /home/mlb-edgar/<folder with hyperspot_project_files> && conda run --no-capture-output -n HyperSpot \
      hyperspot <command> --project <project name> <options> \
      > logs/<log name>.log 2>&1"
```

`hyperspot --help` lists the commands (`run-pipeline`, `train-embedder`, `make-embedding`,
`compute-spatial-domains`, `compute-celltype-spatial-domains`, `compute-ct-svgs`, `perform-model-evaluation`, …).
