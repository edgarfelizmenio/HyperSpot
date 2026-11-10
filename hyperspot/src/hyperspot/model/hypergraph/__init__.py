from __future__ import annotations

import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from pathlib import Path

from ...metadata import ProjectMetadata

@dataclass
class HypergraphEmbedderArgs:
    model_name: str
    input_dim: int
    embedding_dim: int
    dropout: float = 0.2
    hyperedge_feat_dim: Optional[int] = None   # set by the driver from the hypergraph

    
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
    

# --- registry -----------------------------------------------------------------
# Each embedder module registers itself on import (@HypergraphEmbedder.register).
from .cthgcaev6 import CTHGCAEv6Embedder  # noqa: E402,F401

# Optional hook: a local `_local_embedders.py` (not part of the package) may import
# additional embedder modules so they register too.
import importlib as _importlib  # noqa: E402
try:
    _importlib.import_module(f"{__name__}._local_embedders")
except ModuleNotFoundError as e:
    if e.name != f"{__name__}._local_embedders":
        raise
