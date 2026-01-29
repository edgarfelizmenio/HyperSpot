import torch

import torch.nn as nn

class ModuleDict(nn.Module):
    def __init__(self, num_keys, input_dim, output_dim):
        super(ModuleDict, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = nn.Embedding(num_keys, input_dim * output_dim)
        self.bias = nn.Embedding(num_keys, output_dim)
    
    # TODO do LoRA
    def forward(self, X, keys):
        """
        X    : (N, input_dim)
        keys : (N,) or (N, 1)
        """
        keys = keys.squeeze(-1)

        # (N, input_dim * output_dim) → (N, input_dim, output_dim)
        W = self.weight(keys).view(-1, self.input_dim, self.output_dim)

        # (N, 1, input_dim)
        X = X.unsqueeze(1)

        # Batch matmul → (N, 1, output_dim)
        out = torch.bmm(X, W)

        # Remove singleton dim → (N, output_dim)
        out = out.squeeze(1)

        # Add per-key bias
        out = out + self.bias(keys)

        return out

# 3 layers
class MLPDict(nn.Module):
    def __init__(self, num_keys, input_dim, hidden_dim, output_dim):
        super(MLPDict, self).__init__()
        self.hidden_layer = ModuleDict(num_keys, input_dim, hidden_dim)
        self.output_layer = ModuleDict(num_keys, hidden_dim, output_dim)
        self.act = nn.ReLU()
    
    def forward(self, X, keys):
        z = self.hidden_layer(X, keys)
        z = self.act(z)
        out = self.output_layer(z, keys)
        return out
        