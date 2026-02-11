import torch

import torch.nn as nn

class LinearDict(nn.Module):
    def __init__(self, num_keys, input_dim, output_dim, bias=False, rank=50):
        super(LinearDict, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # shared weights
        self.W0 = nn.Linear(input_dim, output_dim, bias=False)

        self.A = nn.Embedding(num_keys, input_dim * rank)
        self.B = nn.Embedding(num_keys, rank * output_dim)

        if bias:
            self.bias = nn.Embedding(num_keys, output_dim)

    def reset_parameters(self):
        self.W0.reset_parameters()
        self.A.reset_parameters()
        self.B.reset_parameters()
        if hasattr(self, 'bias'):
            self.bias.reset_parameters()
    
    # LoRA implementation to handle scale of parameters
    def forward(self, X, keys):
        """
        X    : (N, input_dim)
        keys : (N,) or (N, 1)
        """
        keys = keys.squeeze(-1)

        # Base output: (N, output_dim)
        out = self.W0(X)

        # Lookup & reshape
        # A_k: (N, input_dim, rank)
        # B_k: (N, rank, output_dim)
        A = self.A(keys).view(-1, self.input_dim, self.rank)
        B = self.B(keys).view(-1, self.rank, self.output_dim)

        # LoRA path using einsum
        # X @ A_k → (N, rank)
        z = torch.einsum("nd,ndr->nr", X, A)

        # z @ B_k → (N, output_dim)
        z = torch.einsum("nr,nro->no", z, B)

        out = out + z

        if hasattr(self, 'bias'):
            out = out + self.bias(keys)

        return out

# Example usage on a 3 layer neural network.
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
        