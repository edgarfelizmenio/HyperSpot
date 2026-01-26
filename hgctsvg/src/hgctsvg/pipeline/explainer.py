import numpy as np

import  torch

from dataclasses import dataclass

from pathlib import Path
from tqdm import tqdm
from typing import List

from captum.attr import IntegratedGradients

from ..data import STHypergraphData

from ..model.classifier import EmbeddingClassifier, EmbeddingClassifierArgs
from ..model.hypergraph import HypergraphEmbedder, HypergraphEmbedderArgs

from ..config import AcceleratorOptions
from .utils import get_device

from .classification import load_sc_classifier_input

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

        if not self.output_dir.exists():
            self.output_dir.mkdir()
    
    def compute_spatial_domain_genes(self):
        pass

    def compute_spatial_domain_metagenes(self, cell_type):
        pass

    def compute_cell_type_spatial_embedding_attribution(self, 
                                                   cell_types: List[str] | str | None,
                                                   embedding_dim,
                                                   num_layers,
                                                   args: ExplainerArgs):
        
        self.device = get_device(args.accelerator)

        if cell_types is None:
            cell_types = np.loadtxt(self.project / "cell_types.txt", dtype=str)
        elif isinstance(cell_types, str):
            cell_types = list[cell_types]
        
        with tqdm(cell_types) as ct_pbar:
            for cell_type in ct_pbar:

                ct_pbar.set_postfix({'Cell Type': cell_type})
                # load data
                embeddings, classes, num_classes = load_sc_classifier_input(self.project, cell_type)
                
                classifier_args = EmbeddingClassifierArgs(cell_type, embedding_dim, num_classes, num_layers)
                classifier_model = EmbeddingClassifier(classifier_args)
                classifier_model.load_checkpoint(self.project)

                # TODO: perform batch training
                
                classifier_model.to(self.device)
                embeddings = embeddings.to(self.device)
                classes = classes.to(self.device)
                
                ig = IntegratedGradients(classifier_model)
                attributions = ig.attribute(embeddings, 
                                            target=classes, 
                                            n_steps=args.n_steps,
                                            internal_batch_size=embeddings.shape[0])
        
                # process attributions
                # group by label and compute mean attribution per label
                classes = classes.view(classes.size(0), 1).expand(-1, attributions.size(1))
                unique_classes, classes_count = classes.unique(dim=0, return_counts=True)

                attr_cell_type = {}
                
                total_attr_per_label = torch.zeros_like(unique_classes, 
                                                        dtype=torch.float, 
                                                        device=self.device).scatter_add(0, classes, attributions)
                
                mean_attr_per_label = total_attr_per_label / classes_count.float().unsqueeze(1)

                attributions = attributions.cpu().numpy()
                total_attr_per_label = mean_attr_per_label.cpu().numpy() 
                mean_attr_per_label = mean_attr_per_label.cpu().numpy()

                attr_cell_type["attributions"] = attributions
                attr_cell_type["total_per_label"] = total_attr_per_label
                attr_cell_type["mean_per_label"] = mean_attr_per_label
                np.savez(self.output_dir / f"attributions_{cell_type}.npz", **attr_cell_type) 

    def compute_metagene_to_region_attribution(self,
                                               cell_type: str,
                                               embedding_dim: int,
                                               region: int,
                                               explainer_args: ExplainerArgs):

        self.device = get_device(explainer_args.accelerator)
        
        embeddings, classes, num_classes = load_sc_classifier_input(self.project, cell_type)

        classifier_args = EmbeddingClassifierArgs(cell_type, embedding_dim, num_classes, num_layers)
        classifier_model = EmbeddingClassifier(classifier_args)
        classifier_model.load_checkpoint(self.project)

        classifier_model.to(self.device)
        embeddings = embeddings.to(self.device)
        
        ig = IntegratedGradients(classifier_model)
        attributions = ig.attribute(embeddings, 
                                   target=region, 
                                   n_steps=explainer_args.n_steps,
                                   internal_batch_size=embeddings.shape[0])

        np.savez()
        pass
        
    def compute_cell_feature_attribution(self,
                                         hypergraph: STHypergraphData,
                                         args: HypergraphEmbedderArgs,
                                         embedding_idx: List[int] | int,
                                         batch_size: int,
                                         n_steps: int,
                                         accelerator: AcceleratorOptions):
        
        if isinstance(embedding_idx, int):
            embedding_idx = list(embedding_idx)
        
        self.device = get_device(accelerator)
        self.embedder = HypergraphEmbedder.build(self.project, args)
        self.embedder.load_checkpoint(self.project)
        self.embedder.to(self.device)
        self.embedder.eval()

        forward_fn = self.embedder.get_attribution_forward()
        ig = IntegratedGradients(forward_fn)

        # TODO load dataloder
        
        with tqdm(embedding_idx) as embedding_pbar:
            for idx in embedding_pbar:
                with tqdm(dataloader) as batch_pbar:
                    for batch in batch_pbar:
                        batch = batch.to_device(self.device)
                        inputs, additional_forward_args = self.embedder.get_attribution_inputs(batch)
                        idx_attribution = ig.attribute(inputs,
                                                       target=idx,
                                                       n_steps=n_steps,
                                                       internal_batch_size=inputs.shape[0],
                                                       additional_forward_args = additional_forward_args)

                        
                # attributions = 
                # TODO save attribution score for embedding of each batch
            # TODO save attribution scores for each embedding

    def load_embedding_index_list(self, cell_types=List[str]):
        attributions = {
        }
        for cell_type in cell_types:
            pass
            # attributions[]   

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


