import torch

import torch.nn as nn
import torch.nn.functional as F

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from .cthgnn import CTHyFiHGNNv1
from ..moduledict import LinearDict, ParameterDict

class CTHGCAE(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout=0.1):
        # self.encoder = CTHyFiHGNN()
        pass

