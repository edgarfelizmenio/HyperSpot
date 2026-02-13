import gzip
import numpy as np

import scanpy as sc

import torch
import torch.nn.functional as F

from dataclasses import dataclass

from pathlib import Path
from tqdm import tqdm
from typing import List

from captum.attr import IntegratedGradients, DeepLift

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from ..data import STHypergraphData, get_ids_per_celltype, get_celltype_ids
from ..dataset.hypergraph import STHypergraphMatrixDataset, STHypergraphBatchMatrixData

from ..model.classifier import EmbeddingClassifier, EmbeddingClassifierArgs
from ..model.hypergraph import HypergraphEmbedder, HypergraphEmbedderArgs

from ..config import AcceleratorOptions
from .utils import get_device

from .classification import load_sc_classifier_input
from .utils import get_celltype_mapping

# identify important features
# candidates
#  Integrated Gradients

# The minimum value for internal batch size is the actual batch size,
# so it's easier to automatically set the internal batch size to batch size 
# and not set as CLI args
@dataclass
class ExplainerArgs:
    n_steps: int
    batch_size: int
    accelerator: AcceleratorOptions

class Explainer(object):
    def __init__(self, project):
        self.project = project
        
        self.output_dir = self.project / "attribution"
        self.output_dir.mkdir(exist_ok=True)
            
    def compute_spatial_domain_genes(self):
        raise NotImplementedError()

    def compute_spatial_domain_metagenes(self, cell_type):
        raise NotImplementedError()

    def compute_cell_feature_to_cell_metagene_attribution(self,
                                                hypergraph: STHypergraphData,
                                                embedder_args: HypergraphEmbedderArgs,
                                                explainer_args: ExplainerArgs,
                                                embedding_idx: List[int] | int = None):
        
        self.device = get_device(explainer_args.accelerator)
        self.embedder = HypergraphEmbedder.build(self.project, embedder_args)
        self.embedder.load_checkpoint(self.project)
        self.embedder.to(self.device)
        self.embedder.eval()

        if embedding_idx is None:
            embedding_idx = list(range(embedder_args.embedding_dim))
        elif isinstance(embedding_idx, int):
            embedding_idx = list(embedding_idx)

        hgdataset = STHypergraphMatrixDataset(hypergraph)

        forward_fn = self.embedder.get_attribution_forward()
        ig = IntegratedGradients(forward_fn)

        # loop structure:
        # embedding index -> celltype -> mini batch 
        with tqdm(embedding_idx, position=0) as embedding_pbar:
            for idx in embedding_pbar:        
                # compute attributions per celltype.
                # order processing based on increasing batch size given a celltype
                idx_attribution_ct_batches = []
                idx_attribution_ct_ids = []
                
                processed_cells = torch.zeros(len(hgdataset), dtype=torch.bool).to(self.device)
                # print(processed_cells.shape)
                per_celltype_idx = get_ids_per_celltype(hypergraph)
                total_processed = 0
                i = 0
                processed_pbar = tqdm(total = len(hgdataset), leave=False, position=1)
                while not processed_cells.all():
                    # pick celltype with smallest batch
                    i = min((j for j in range(len(per_celltype_idx)) if per_celltype_idx[j].numel() > 0), 
                            key=lambda j: hgdataset[per_celltype_idx[j].tolist()].batch_subgraph_cells.shape[0])
            
                    # get ids of cells of the sample and the neighbors
                    ct_batch_sample = per_celltype_idx[i]
                    ct_batch_ids = hgdataset[ct_batch_sample.tolist()].batch_subgraph_cells.to(self.device)
                    
                    # perform attribution computation
                    hgexpsampler = BatchSampler(
                        sampler = ct_batch_sample.tolist(),
                        batch_size = explainer_args.batch_size,
                        drop_last=False
                    )
            
                    hgloader = DataLoader(hgdataset, sampler=hgexpsampler, batch_size=None)
                    
                    idx_attribution_batches = []   
                    total_processed = 0
                    with tqdm(hgloader, leave=False, position=2) as batch_pbar:
                        for batch in batch_pbar:
                            batch = batch.to_device(self.device)

                            # find the unprocessed cells
                            is_batch_cell_unprocessed = ~processed_cells[batch.batch_subgraph_cells]
                            unprocessed_batch_cells = batch.batch_subgraph_cells[is_batch_cell_unprocessed]
                            
                            inputs, additional_forward_args = self.embedder.get_attribution_inputs(batch)
                            
                            idx_attribution_batch = ig.attribute(inputs,
                                                           target=idx,
                                                           n_steps=explainer_args.n_steps,
                                                           internal_batch_size=inputs.shape[0],
                                                           additional_forward_args = additional_forward_args)

                            # dont store the cells that are already processed on previous iterations to prevent duplicates 
                            idx_attribution_batch = idx_attribution_batch[is_batch_cell_unprocessed]
                            idx_attribution_batches.append(idx_attribution_batch)

                            # mark unprocessed batch cells as processed
                            idx_attribution_ct_ids.append(unprocessed_batch_cells)
                            processed_cells[unprocessed_batch_cells] = True
                            total_processed += unprocessed_batch_cells.numel()
                    
                    idx_attribution_ct_batch = torch.cat(idx_attribution_batches, dim=0)

                    # print(idx_attribution_ct_batch.shape[0], total_processed, per_celltype_idx[i].numel())
                    # print()
                    # assert idx_attribution_ct_batch.shape[0] == per_celltype_idx[i].numel()
                    assert idx_attribution_ct_batch.shape[0] == total_processed
                    
                    idx_attribution_ct_batches.append(idx_attribution_ct_batch)

                    mask = ~processed_cells.cpu()
                    for j in range(len(per_celltype_idx)):
                        ids = per_celltype_idx[j]
                        per_celltype_idx[j] = ids[mask[ids]]
            
                    # print(i, batch_sample.numel(), batch_ids.numel(), unprocessed.numel(), total)
                    processed_pbar.update(total_processed)
                    processed_pbar.set_postfix({"cell type": i, "processed samples": total_processed})
                
                processed_pbar.close()

                idx_attribution = torch.cat(idx_attribution_ct_batches, dim=0)

                # sort result in increasing node id
                idx_attribution_ids = torch.cat(idx_attribution_ct_ids, dim=0)
                idx_attribution_ids = torch.argsort(idx_attribution_ids)
                idx_attribution = idx_attribution[idx_attribution_ids]
                idx_attribution = F.normalize(idx_attribution)
                idx_attribution = idx_attribution.cpu().numpy()
                self.save_node_embedding_attr_result(idx_attribution, idx)        

    def compute_g2ctd_attribution(self, 
                          cell_type,
                          hypergraph: STHypergraphData,
                          embedder_args: HypergraphEmbedderArgs,
                          num_layers,
                          explainer_args: ExplainerArgs):

        self.device = get_device(explainer_args.accelerator)
        self.embedder = HypergraphEmbedder.build(self.project, embedder_args)
        self.embedder.load_checkpoint(self.project)
        self.embedder.to(self.device)
        self.embedder.eval()

        _, cell_type_mapping = get_celltype_mapping(self.project)
        ct_idx = cell_type_mapping[cell_type]
                
        hgdataset = STHypergraphMatrixDataset(hypergraph)
        _, classes, num_classes = load_sc_classifier_input(self.project, cell_type)
        ids = get_celltype_ids(hypergraph, ctidx)

        classifier_args = EmbeddingClassifierArgs(cell_type, embedder_args.embedding_dim, num_classes, num_layers)
        self.classifier = EmbeddingClassifier(classifier_args)
        self.classifier.load_checkpoint(self.project)
        self.classifier.to(self.device)
        self.classifier.eval()

        forward_fn = self.embedder.get_e2e_attribution_forward(self.classifier, num_repeats=1)
        ig = IntegratedGradients(forward_fn, multiply_by_inputs=True)
        
        dataset = TensorDataset(ids, classes)
        loader = DataLoader(dataset, batch_size=explainer_args.batch_size, shuffle=False, drop_last=False)
            
        attribution_batches = []
        with tqdm(loader, leave=False, position=0) as batch_pbar:
            for batch_ids, batch_classes in batch_pbar:
                print(batch_classes.shape)
                hg_batch = hgdataset[batch_ids.tolist()]

                hg_batch = hg_batch.to_device(self.device)
                inputs, additional_forward_args = self.embedder.get_e2e_attribution_inputs(hg_batch)        
                # inputs = inputs.to(self.device)
                # incidence_matrix = additional_forward_args[0].to(self.device)
                # batch_size = additional_forward_args[1]
                batch_classes = batch_classes.to(self.device)
                attribution_batch = ig.attribute(inputs, 
                                            target=batch_classes, 
                                            # n_steps=explainer_args.n_steps,
                                            internal_batch_size=inputs.shape[0],
                                            additional_forward_args = additional_forward_args)
                                                            
                # attribution_batch = F.normalize(attribution_batch)

                attribution_batch = attribution_batch[:hg_batch.sample_n_cells]
                attribution_batches.append(attribution_batch)
        
        attributions = torch.cat(attribution_batches, dim=0)

        assert attributions.shape[0] == ids.shape[0]
        assert attributions.shape[0] == classes.shape[0]
        
        # process attributions
        # group by label and compute mean attribution per label
        classes = classes.view(classes.size(0), 1).expand(-1, attributions.size(1)).to(attributions.device)
        unique_classes, classes_count = classes.unique(dim=0, return_counts=True)

        attr_cell_type = {}

        total_attr_per_label = torch.zeros_like(unique_classes, 
                                                dtype=torch.float, 
                                                device=self.device).scatter_add(0, classes, attributions)
        
        mean_attr_per_label = total_attr_per_label / classes_count.float().unsqueeze(1)

        attributions = attributions.detach().cpu().numpy()

        self.save_ct_result(cell_type, attributions)
        
    def save_ct_result(self, cell_type, result_data: np.ndarray):
        ct_results_dir = self.output_dir / f"cell_type_{cell_type}"
        ct_results_dir.mkdir(exist_ok=True)

        np.save(ct_results_dir / f"attributions.npy", result_data)

    def load_ct_results(self, cell_type):
        ct_results_dir = self.output_dir / f"cell_type_{cell_type}"
        return np.load(ct_results_dir / f"attributions.npy")

    def find_domain_gene_idx(self, cell_type, pct=95):
        attributions = self.load_ct_results(cell_type)
        
        adata_celltype = sc.read_h5ad(self.project / "clustering" / f"ST_cell_{cell_type}.h5ad")
        num_regions = adata_celltype.obs['leiden'].nunique()
        regions = adata_celltype.obs['leiden'].to_numpy().astype(int)
        
        return find_relevant_features(attributions, regions, num_regions, pct)


def find_relevant_features(attributions, indexes, num_indexes, pct=95):
    """
    Find metagenes associated with cell spatial domains.
    """
    attr_grps = [attributions[indexes == index] for index in range(num_indexes)]
    
    pos_attr_grps = [np.where(attr_grps[i] > 0, attr_grps[i], 0) for i in range(num_indexes)]
    neg_attr_grps = [np.where(attr_grps[i] < 0, attr_grps[i], 0) for i in range(num_indexes)]
    
    # set negative attributions as zero
    pos_feat_idx, pos_attr_norms = find_impt_feature_idx(pos_attr_grps, num_indexes, pct)
    neg_feat_idx, neg_attr_norms = find_impt_feature_idx(neg_attr_grps, num_indexes, pct)

    return pos_feat_idx, neg_feat_idx, pos_attr_norms, neg_attr_norms

def find_impt_feature_idx(attr_grps, num_groups, pct=95):
    # compute norm for each group to summarize overall attribution of feature accross all samples in the classification
    attr_grp_norms = [np.linalg.norm(attr_grps[i], axis=0) for i in range(num_groups)]    
    threshold = [np.percentile(attr_grp_norms[i], pct) for i in range(num_groups)]
    
    for i in range(num_groups):
        print(i, threshold[i])
        print(attr_grp_norms[i])
    # attr_grp_mean = [np.mean(attr_grp_norms[i]) for i in range(num_groups)]
    # attr_grp_std = [np.std(attr_grp_norms[i]) for i in range(num_groups)]

    # Features with attribution norm higher than mean + alpha*std will be considered important
    # attr_grp_impt = [attr_grp_norms[i] > attr_grp_mean[i] + (alpha * attr_grp_std[i]) for i in range(num_groups)]

    # TODO categorize
    # high pos low neg -> positive attribution
    # high pos high neg -> case by case
    # high neg low pos -> negative attribution
    # log neg low pos -> irrelevant

    
    # eliminate genes with low variance across groups

    attr_grp_impt = [attr_grp_norms[i] > threshold[i] for i in range(num_groups)]
    attr_grp_impt_feat_idx = [attr_grp_impt[i].nonzero()[0] for i in range(num_groups)]

    # order by decreasing norm of attributions
    attr_grp_impt_feat_idx_order = [np.argsort(-attr_grp_norms[i][attr_grp_impt_feat_idx[i]]) for i in range(num_groups)]
    attr_grp_impt_feat_idx = [attr_grp_impt_feat_idx[i][attr_grp_impt_feat_idx_order[i]] for i in range(num_groups)]
    
    return attr_grp_impt_feat_idx, attr_grp_norms
    


def find_representative_embeddings(embeddings):
    # compute centroid (mean embedding)
    embedding_centroid = embeddings.mean(axis=0)

    # compute cosine similarity of centroid against all vectors
    # and  find the most similar vector
    embeddings_norm = np.linalg.norm(embeddings, axis=1)
    centroid_norm = np.linalg.norm(embedding_centroid)

    similarities = centroid_norm @ embeddings_norm.T
    most_similar_index = np.argmax(similarities)

    representative_embedding = embeddings[most_similar_index]
    
    return representative_embedding
