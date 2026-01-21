import numpy as np

import  torch

from pathlib import Path
from typing import List

from captum.attr import IntegratedGradients

from ..data import STHypergraphData

from ..model.classifier import EmbeddingClassifier
from ..model.hypergraph import HypergraphEmbedder, HypergraphEmbedderArgs

from ..config import AcceleratorOptions
from .utils import get_device

# identify important features
# candidates
#  Integrated Gradients
#  DeepLift
#  Feature Ablation

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
                                                   cell_type,
                                                   embedding_dim,
                                                   num_layers,
                                                   num_classes,
                                                   embeddings,
                                                   labels,
                                                   n_steps,
                                                   # batch_size: int,
                                                   accelerator: AcceleratorOptions):
        self.device = get_device(accelerator)

        classifier_model = EmbeddingClassifier(embedding_dim, num_classes, num_layers)
        model_checkpoint = torch.load(self.project / 'classifiers' / f'cell_{cell_type}.pth', weights_only=True)
        classifier_model_state = model_checkpoint['model_state_dict']
        classifier_model.load_state_dict(classifier_model_state)

        # TODO: perform batch training
        classifier_model.to(self.device)
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        
        ig = IntegratedGradients(classifier_model)
        attributions = ig.attribute(embeddings, 
                                   target=labels, 
                                   n_steps=n_steps,
                                   internal_batch_size=embeddings.shape[0])

        # process attributions
        # group by label and compute mean attribution per label
        labels = labels.view(labels.size(0), 1).expand(-1, attributions.size(1))
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        total_attr_per_label = torch.zeros_like(unique_labels, dtype=torch.float, device=self.device).scatter_add(0, labels, attributions)
        mean_attr_per_label = total_attr_per_label / labels_count.float().unsqueeze(1)
        mean_attr_per_label = mean_attr_per_label.cpu().numpy()
        
        np.save(self.output_dir / f"attributions_{cell_type}.npy", mean_attr_per_label) 
     
    def compute_cell_type_spatial_embedding_component_attribution(self,
                                                                  hypergraph: STHypergraphData,
                                                                  args: HypergraphEmbedderArgs,
                                                                  component_idx: List[int],
                                                                  batch_size: int,
                                                                  n_steps: int,
                                                                  attr_batch_size: int,
                                                                  accelerator: AcceleratorOptions):

        self.device = get_device(accelerator)
        self.embedder = HypergraphEmbedder.build(self.project, args)
        self.embedder.load_checkpoint(self.project)
        self.embedder.to(self.device)
        self.embedder.eval()

        forward_fn = self.embedder.get_attribution_forward()

        with tqdm(component_idx) as component_pbar:
            for idx in componend_pbar:
                with tqdm(dataloader) as batch_pbar:
                    for batch in batch_pbar:
                        batch = batch.to_device(self.device)
                        inputs, additional_forward_args = self.embedder.get_attribution_inputs(batch)
                        attr_idx = ig.attribute(inputs,
                                                target=idx,
                                                n_steps=n_steps,
                                                internal_batch_size=attr_batch_size,
                                                additional_forward_args = additional_forward_args)
                # TODO save attribution score for component of each batch
            # TODO save attribution scores for each component

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


