# HyperSpot

HyperSpot: Detection of Cell Type-Specific Spatially Variable Gene Modules using Spatially-Informed Hypergraph Embeddings.

## Changelog

### 2026-06-17

#### Bug Fixes
- **`dataset/hypergraph/matrix_dataset.py`**: Fixed `.to(torch.bool)` cast on incidence matrix that destroyed float intra/inter-spot weights (5.0 and 1.0 both became 1.0); matrix is now kept as float with `.bool()` applied inline only where boolean logic is required

#### Features
- **`preprocessing/visium.py`**: `create_hypergraph_data` accepts `intra_weight` and `inter_weight` parameters for weighted intra/inter-spot hyperedges (previously binary)
- **`pipeline/clustering.py`**: PCA before KNN/Leiden (default 50 components, capped at `min(embedding_dim-1, n_obs-1)`); connectivity-based cluster merging as new default (`merge_mode="connectivity"`) — merges undersized clusters into their most graph-connected neighbor, consistent with Leiden community structure; centroid and per-cell modes retained
- **`cli.py`**: `ClusterArgs` gains `n_pca_components: int = 50` CLI parameter

#### Refactoring
- **`model/hypergraph/__init__.py`**: `HypergraphEmbedder.build(args, n_cell_types)` takes `n_cell_types` explicitly from pipeline and passes to all `_build_impl` implementations
- **`pipeline/embedder.py`**: `__setup_model` reads `ProjectMetadata().n_cell_types` and passes to `build()`; hardcoded `n_classes=12` in `fit()` replaced with `project_metadata.n_cell_types`
- **`pipeline/explainer.py`**: `HypergraphEmbedder.build` call updated to pass `project_metadata.n_cell_types`

## Installation

### Production Mode
Install HyperSpot package using pip:
  ```bash
  pip install hyperspot
  ```
  
### Development Mode
Install in editable mode using pip:
  ```bash
  pip install -e hyperspot
  ```

## Reproducibility

HyperSpot seeds its random number generators once, at the entry point, never inside library functions
(`hyperspot.utils.set_seed` seeds python, numpy, torch CPU/CUDA and makes cuDNN deterministic).

- **CLI:** every command is a separate program and seeds itself at startup (default seed 42).
- **`perform-model-evaluation`:** each run reseeds with its own fixed seed before
  train → embed → cluster → score, so run *i* depends only on its seed (also when resumed).
- **Notebooks / Python API:** call `set_seed(seed)` once before running the pipeline steps, and keep the
  same step order. Without it, `HypergraphEmbedderDriver.fit` (initialisation, batch order) and
  `generate_embedding` (K/M neighbourhood sampling) are not reproducible.
- Leiden clustering takes its own `seed` argument (default 42).

Bitwise reproducibility also assumes the same GPU, library versions and batch size.
