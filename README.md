# HG ct-SVG

HGCTSVG: Detection of Cell Type-Specific Spatially Variable Gene Modules using Spatially-Informed Hypergraph Embeddings.

## Changelog

### 2026-06-17

#### Bug Fixes
- **`dataset/hypergraph/matrix_dataset.py`**: Fixed `.to(torch.bool)` cast on incidence matrix that destroyed float intra/inter-spot weights (5.0 and 1.0 both became 1.0); matrix is now kept as float with `.bool()` applied inline only where boolean logic is required

#### Features
- **`preprocessing/visium.py`**: `create_hypergraph_data` accepts `intra_weight` and `inter_weight` parameters for weighted intra/inter-spot hyperedges (previously binary)
- **`pipeline/clustering.py`**: PCA before KNN/Leiden (default 50 components, capped at `min(embedding_dim-1, n_obs-1)`); connectivity-based cluster merging as new default (`merge_mode="connectivity"`) — merges undersized clusters into their most graph-connected neighbor, consistent with Leiden community structure; centroid and per-cell modes retained
- **`cli.py`**: `ClusterArgs` gains `n_pca_components: int = 50` CLI parameter

#### Refactoring
- **`model/hypergraph/cthgnn.py`**: `CTHyFiHGNNv1` `n_cell_types` is now a constructor parameter (replaces hardcoded 12 in `LinearDict` and `ParameterDict`); node input dropout disabled
- **`model/hypergraph/cthgcae.py`**: `CTHGCAE` accepts and forwards `n_cell_types` to both encoder and decoder
- **`model/hypergraph/cthgcaev2.py`**: `CTHGCAEv2` forwards `num_cell_types` to encoder; `CTHGCAEv2Embedder` uses `n_cell_types` from pipeline (removed try/except fallback to 12)
- **`model/hypergraph/__init__.py`**: `HypergraphEmbedder.build(args, n_cell_types)` takes `n_cell_types` explicitly from pipeline and passes to all `_build_impl` implementations
- **`model/hypergraph/hgnn.py`, `fusion.py`, `ctfusion.py`, `transformer.py`**: `_build_impl` signatures updated to accept `n_cell_types` for interface consistency
- **`pipeline/embedder.py`**: `__setup_model` reads `ProjectMetadata().n_cell_types` and passes to `build()`; hardcoded `n_classes=12` in `fit()` replaced with `project_metadata.n_cell_types`
- **`pipeline/explainer.py`**: `HypergraphEmbedder.build` call updated to pass `project_metadata.n_cell_types`

## Installation

### Production Mode
Install hgctsvg package using pip:
  ```bash
  pip install hgctsvg
  ```
  
### Development Mode
Install in editable mode using pip:
  ```bash
  pip install -e hgctsvg
  ```

## Repository Management
### Commiting to submodule (STAMarker, SpatialScope)
1. Go to submodule directory (ex. for STAMarker)
  ```bash
  cd STAMarker
  ```
2. Commit the changes in the submodule
  ```bash
  git add <changed files>
  git commit -m "commit message"
  git push
  ```
### Updating the main project to reflect the submodule commit
1. Go to the root of the project directory
2. Update submodule to the latest commit
  ```bash
  git add STAMarker
  git commit -m "commit message"
  git push
  ```
### Pulling changes
1. Pull the parent repository
  ```bash
  git pull --recurse-submodules
  ```
3. Update the submodule itself to match the new pointer
  ```bash
  git submodule update --init --recursive
  ```
#### Shorthand method
  ```bash
  git pull --recurse-submodules --jobs=4
  ```