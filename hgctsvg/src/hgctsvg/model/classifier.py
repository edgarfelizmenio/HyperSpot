import torch
import torch.nn as nn

from dataclasses import dataclass
from pathlib import Path

from .utils import compute_fc_layer_sizes, create_fc

@dataclass
class EmbeddingClassifierArgs:
    cell_type: str
    input_dim: int
    n_classes: int
    n_layers: int = 3

class EmbeddingClassifier(nn.Module):
    def __init__(self, args: EmbeddingClassifierArgs):
        super(EmbeddingClassifier, self).__init__()
        self.cell_type = args.cell_type
        self.input_dim = args.input_dim
        self.n_classes = args.n_classes
        
        layer_sizes = compute_fc_layer_sizes(self.input_dim, self.n_classes, num_layers = args.n_layers)
        
        self.backbone_fc = create_fc(layer_sizes[:-1], bias=True)
        self.out_fc = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, X):
        h = self.backbone_fc(X)
        out = self.out_fc(h)
        return out

    def load_checkpoint(self, project: Path):
        model_checkpoint = torch.load(project / 'classifiers' / self.checkpoint_name,
                                      weights_only=True)
        classifier_model_state = model_checkpoint['classifier_state_dict']
        self.load_state_dict(classifier_model_state)

    @property
    def checkpoint_name(self) -> str:
        return f"cell_{self.cell_type}.pth"