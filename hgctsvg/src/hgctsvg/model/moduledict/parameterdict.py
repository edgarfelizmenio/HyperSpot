import torch

import torch.nn as nn

class ParameterDict(nn.Module):
    def __init__(self, num_keys, output_dim):
        super(ParameterDict, self).__init__()

        self.output_dim = output_dim
        self.params = nn.Embedding(num_keys, output_dim)

    def reset_parameters():
        self.params.reset_parameters()
    
    def forward(self, keys):
        """
        keys : (N,) or (N, 1)
        """
        keys = keys.squeeze(-1)
        return self.params(keys)        