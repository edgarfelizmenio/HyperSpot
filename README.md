# HG ct-SVG

HGCTSVG: Detection of Cell Type-Specific Spatially Variable Gene Modules using Spatially-Informed Hypergraph Embeddings.

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