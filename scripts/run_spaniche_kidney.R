#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# SpaNiche baseline runner (#14) — kidney cancer, one section.
#
# Reads three pre-exported, barcode-aligned inputs (produced by the notebook's
# Python export cell) and runs SpaNiche's single-sample joint-NMF of the
# neighbour cell-type-abundance view + the ligand-receptor view, following the
# author tutorial (tutorial/1a.singlesample_iNMF.md):
#   spaniche_nmf_celltype_and_lr(topic_num=15, defined_weight=c(0.6,0.4),
#                                lambda_v=0.5, sigma_v=0.5)
# The NMF spot x topic loading W is written out; the notebook takes argmax over
# topics per spot = the SpaNiche "ecotype" for the ARI vs HyperSpot communities.
#
# Inputs (all in <indir>, rownames/colnames = the SAME spot barcodes):
#   counts_lognorm.csv   genes(symbol) x spots, log-normalised expression (MT-free)
#   spatialdf.csv        spot x {row,col}; col already = array_col/sqrt(3) (hex)
#   spot_by_celltype.csv spot x cell-type proportions (Spotiphy), 12 CTs
# Output: <indir>/spaniche_spot_topic.csv  (spot x topic W, + argmax 'ecotype')
#
# Usage: Rscript run_spaniche_kidney.R <indir> [topic_num]
# ---------------------------------------------------------------------------
for (v in c("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"))
  do.call(Sys.setenv, setNames(list("4"), v))   # be considerate of shared compute

args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 1) stop("usage: run_spaniche_kidney.R <indir> [topic_num]")
indir <- args[[1]]
topic_num <- if (length(args) >= 2) as.integer(args[[2]]) else 15L

suppressMessages({library(SpaNiche); library(Seurat); library(tidyverse)})
options(Seurat.object.assay.version = "v3")     # SpaNiche assumes v3 Assay (@scale.data)
set.seed(0)

## ---- load barcode-aligned inputs ----------------------------------------
counts <- read.csv(file.path(indir,"counts_lognorm.csv"), row.names=1, check.names=FALSE)
counts <- as.matrix(counts)                                  # genes x spots
spatialdf <- read.csv(file.path(indir,"spatialdf.csv"), row.names=1, check.names=FALSE)
spot_by_celltype <- read.csv(file.path(indir,"spot_by_celltype.csv"), row.names=1, check.names=FALSE)

## enforce a single consistent spot order everywhere
spots <- intersect(intersect(colnames(counts), rownames(spatialdf)), rownames(spot_by_celltype))
cat(sprintf("aligned spots: %d | genes: %d | cell types: %d\n",
            length(spots), nrow(counts), ncol(spot_by_celltype)))
counts <- counts[, spots]
spatialdf <- spatialdf[spots, c("row","col"), drop=FALSE]
spot_by_celltype <- spot_by_celltype[spots, , drop=FALSE]

## ---- Seurat object (log-normalised already => data slot) + hex image -----
seu <- CreateSeuratObject(counts, assay="Spatial")
seu[["Spatial"]]@data <- counts                              # already log1p-normalised
coords <- data.frame(x=spatialdf$row, y=spatialdf$col, row.names=spots)
seu[["slice"]] <- new("SlideSeq", assay="Spatial", coordinates=coords, key="slice_")

## ---- run SpaNiche joint NMF (celltype + LR views) ------------------------
setwd(indir)                                                 # SpaNiche writes byproducts to cwd
res <- spaniche_nmf_celltype_and_lr(
  spatial.seu = seu, spatialdf = spatialdf, spot_by_celltype = spot_by_celltype,
  smoothing_type = c("view0","view1","view2"),
  distance_thre  = c(NA, 2/(3^0.5), 4/(3^0.5)),
  LR_dm_type     = c("view1","view2"),
  LR_distance_thre = c(2/(3^0.5), 4/(3^0.5)),
  topic_num = topic_num, defined_weight = c(0.6,0.4),
  lambda_v = 0.5, sigma_v = 0.5)

W <- as.data.frame(res$nmf_res$W)                            # spot x topic
colnames(W) <- paste0("topic", seq_len(ncol(W)))
rownames(W) <- spots
W$ecotype <- max.col(as.matrix(W), ties.method="first")      # argmax topic = ecotype
write.csv(W, file.path(indir,"spaniche_spot_topic.csv"), quote=FALSE)
cat(sprintf("wrote spaniche_spot_topic.csv: %d spots, %d topics, %d ecotypes used\n",
            nrow(W), ncol(W)-1L, length(unique(W$ecotype))))
