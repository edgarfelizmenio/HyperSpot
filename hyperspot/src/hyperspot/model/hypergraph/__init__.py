from __future__ import annotations

import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Callable, Tuple

from pathlib import Path

from ...metadata import ProjectMetadata

@dataclass
class HypergraphEmbedderArgs:
    model_name: str
    input_dim: int
    embedding_dim: int
    dropout: float = 0.2
    latent_dim: Optional[int] = None
    num_heads: Optional[int] = None
    num_blocks: Optional[int] = None
    num_hyperedges: Optional[int] = None
    hyperedge_feats: Optional[str] = None
    hyperedge_feat_dim: Optional[int] = None
    hyperedge_pool: str = "spot"          # "spot" (own-spot cells) | "neighborhood"

    
class HypergraphEmbedder(nn.Module, ABC):
    _registry: Dict[str, HypergraphEmbedder] = {}
    
    def __init__(self, model: nn.Module):
        super(HypergraphEmbedder, self).__init__()
        self.model = model
        
    def forward(self, batch):
        raise NotImplementedError

    def load_checkpoint(self):
        project_metadata = ProjectMetadata()
        path = project_metadata.project_folder / f'{self.checkpoint_name()}.pth'
        state = torch.load(path, weights_only=True)
        # strict=False: a model trained with the opt-in hyperedge decoder carries
        # extra hyperedge_decoder.* keys that inference doesn't rebuild (its output
        # is discarded at embedding time); tolerate those rather than error.
        self.load_state_dict(state['embedder_state_dict'], strict=False)
    
    @classmethod
    def register(cls, name: str):
        def decorator(subclass: HypergraphEmbedder):
            if name in cls._registry:
                raise KeyError(f"Duplicate embedder name: {name}")
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_available_embedders(cls):
        return list(cls._registry.keys())
    
    @classmethod
    def build(cls, args: HypergraphEmbedderArgs, n_cell_types: int) -> HypergraphEmbedder:
        try:
            subcls = cls._registry[args.model_name]
        except KeyError:
            raise ValueError(f"Unknown model: {args.model_name}")

        embedder_model = subcls._build_impl(args, n_cell_types)

        return embedder_model

    @classmethod
    @abstractmethod
    def checkpoint_name(cls) -> str:
        pass
    
    @abstractmethod
    def get_attribution_forward(self, index = 0) -> Callable[..., torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_attribution_inputs(self, batch) -> Tuple[torch.Tensor, tuple]:
        raise NotImplementedError

    @abstractmethod
    def get_e2e_attribution_forward(self, classifier, index = 0):
        raise NotImplementedError
    
    @abstractmethod
    def get_e2e_attribution_inputs(self, batch):
        raise NotImplementedError

from .fusion import SpatialFusionEmbedder
from .hgnn import HyFiEmbedder
from .transformer import HGTransformerEmbedder
from .cthgnn import CTHyFiEmbedder
from .cthgcae import CTHGCAEEmbedder
from .cthgcaev2 import CTHGCAEv2Embedder
from .cthgcaev3 import CTHGCAEv3Embedder
from .cthgcaev4 import CTHGCAEv4Embedder
from .cthgcaev5 import CTHGCAEv5Embedder
from .cthgcaev6 import CTHGCAEv6Embedder
from .cthgcaev7 import CTHGCAEv7Embedder
