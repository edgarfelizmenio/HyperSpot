#!/usr/bin/env python
"""Export barcode-aligned SpaNiche inputs for one kidney-cancer section (#14).

Writes three CSVs (all keyed by the SAME spot barcodes) into <outdir>:
  counts_lognorm.csv    genes(symbol) x spots, log-normalised, MT-free (from HyperSpot ST_aggregated)
  spatialdf.csv         spot x {row,col}; col = array_col/sqrt(3)  (Visium hex geometry)
  spot_by_celltype.csv  spot x 12 Spotiphy cell-type proportions

Sources (same MT-free proportions/data HyperSpot uses, for a fair baseline):
  ST_aggregated.h5ad         log-normalised, MT-free expression (ENSG)
  ST_decomposition.h5ad      ENSG->SYMBOL map
  <section>.h5ad             array_row/array_col
  ctSVG-benchmarking ST_prop.csv + ST_spots.txt   cell-type proportions (barcode-aligned)

Usage: export_spaniche_inputs.py <Section> <outdir>
"""
import sys, os
import numpy as np, pandas as pd, anndata as ad

SEC = sys.argv[1] if len(sys.argv) > 1 else "Interface"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/spaniche_{SEC}"
os.makedirs(OUT, exist_ok=True)

DEMO = "/workspace/HyperSpot/demo"
PROJ = f"{DEMO}/hyperspot_project_files/Human Kidney Cancer {SEC}"
SPOTI = f"{DEMO}/spotiphy_project_files/Human Kidney Cancer {SEC}/Human Kidney Cancer {SEC}.h5ad"
CTS = f"/workspace/ctSVG-benchmarking/data/4_Human_Kidney_Cancer_{SEC}"

# ---- log-normalised MT-free expression (ENSG -> SYMBOL) --------------------
agg = ad.read_h5ad(f"{PROJ}/ST_aggregated.h5ad")
dec = ad.read_h5ad(f"{PROJ}/ST_decomposition.h5ad", backed="r")
e2s = {e: s for e, s in zip(dec.var_names.astype(str), dec.var["SYMBOL"].astype(str))}
sym = pd.Index([e2s.get(e, e) for e in agg.var_names.astype(str)])
X = np.asarray(agg.X.todense()) if hasattr(agg.X, "todense") else np.asarray(agg.X)
expr = pd.DataFrame(X, index=agg.obs_names.astype(str), columns=sym)      # spots x genes
expr = expr.loc[:, ~expr.columns.duplicated()]                            # keep first per symbol
expr = expr.loc[:, [g for g in expr.columns if not str(g).startswith("ENSG")]]  # mapped symbols only

# ---- array coords -> hex spatialdf (row, col=array_col/sqrt3) --------------
sp = ad.read_h5ad(SPOTI, backed="r")
coord = sp.obs[["array_row", "array_col"]].copy()
coord.index = coord.index.astype(str)
spatialdf = pd.DataFrame({"row": coord["array_row"].astype(float),
                          "col": coord["array_col"].astype(float) / np.sqrt(3)}, index=coord.index)

# ---- cell-type proportions (barcode-aligned) ------------------------------
bc = [l.strip() for l in open(f"{CTS}/ST_spots.txt")]
prop = pd.read_csv(f"{CTS}/ST_prop.csv")
prop.index = pd.Index(bc)

# ---- align on shared barcodes ---------------------------------------------
spots = sorted(set(expr.index) & set(spatialdf.index) & set(prop.index))
expr = expr.loc[spots]; spatialdf = spatialdf.loc[spots]; prop = prop.loc[spots]
print(f"[{SEC}] aligned spots={len(spots)} genes={expr.shape[1]} cell_types={prop.shape[1]}")

expr.T.to_csv(f"{OUT}/counts_lognorm.csv")          # genes x spots
spatialdf.to_csv(f"{OUT}/spatialdf.csv")
prop.to_csv(f"{OUT}/spot_by_celltype.csv")
print("wrote:", OUT)
