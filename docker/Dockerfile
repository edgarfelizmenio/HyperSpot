# HyperSpot image (edgarfelizmenio/hyperspot:1.0). Build from the HyperSpot repo root — see docker/README.md.
FROM condaforge/miniforge3:latest

SHELL ["/bin/bash", "-c"]

# ── HyperSpot conda env (what the package uses, pinned to the versions behind the current results) ──
WORKDIR /opt/setup
COPY docker/HyperSpot_environment.yml .
RUN mamba env create -f HyperSpot_environment.yml -y && conda clean -afy

# ── HyperSpot package from this repo (code as of the build) ──
COPY hyperspot /opt/hyperspot
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate HyperSpot && \
    pip install --no-deps -e /opt/hyperspot && \
    hyperspot --help > /dev/null && \
    python -c "import torch, scanpy, squidpy, igraph, leidenalg, hyperspot.cli, hyperspot.pipeline, hyperspot.preprocessing.visium"

WORKDIR /workspace
