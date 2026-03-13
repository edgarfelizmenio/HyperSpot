import torch

import torch.nn as nn

class LinearDict(nn.Module):
    def __init__(self, num_keys, input_dim, output_dim, bias=False):
        super(LinearDict, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_keys = num_keys

        # shared weights
        self.W = nn.Embedding(num_keys, input_dim * output_dim)

        if bias:
            self.bias = nn.Embedding(num_keys, output_dim)

    def reset_parameters(self):
        self.W.reset_parameters()

        gain = 0.7
        
        for k in range(self.num_keys):
            W = self.W.weight.data[k].view(self.input_dim, self.output_dim)
            nn.init.kaiming_uniform_(W, a=0, mode='fan_in', nonlinearity='linear')
            W.mul_(gain)
                    
        if hasattr(self, 'bias'):
            self.bias.reset_parameters()
            nn.init.zeros_(self.bias.weight)
    
    def forward(self, X, keys):
        """
        X    : (N, input_dim)
        keys : (N,) or (N, 1)
        """
        # Base output: (N, output_dim)   
        orig_shape = X.shape[:-1]

        X_flat = X.reshape(-1, self.input_dim)
        keys_flat = keys.reshape(-1)

        out_flat = torch.zeros((X_flat.shape[0], self.output_dim), device=X.device)
        unique_keys = keys_flat.unique()

        for key in unique_keys:
            idx = (keys_flat == key)
            X_key = X_flat[idx]

            W = self.W.weight[key].view(self.input_dim, self.output_dim)
            out_flat[idx] = out_flat[idx] + X_key @ W

            if hasattr(self, 'bias'):
                b = self.bias.weight[key]
                out_flat[idx] = out_flat[idx] + b

        out = out_flat.reshape(*orig_shape, self.output_dim)

        return out

        # out = torch.zeros((X.shape[0], self.output_dim), device=X.device)
        # keys = keys.squeeze(-1)
        # unique_keys = keys.unique()

        # for key in unique_keys:
        #     idx = (keys == key)
        #     X_key = X[idx] 

        #     print(X_key.shape)
        #     # Lookup & reshape
        #     # W_k: (N, input_dim, output_dim)
        #     W = self.W.weight[key].view(self.input_dim, self.output_dim)

        #     out[idx] = out[idx] + (X_key @ W)

        #     if hasattr(self, 'bias'):
        #         b = self.bias.weight[key]
        #         out[idx] = out[idx] + b

        # return out

# class LinearDictSimple(nn.Module):
#     def __init__(self,  num_keys, input_dim, output_dim, bias=False):
#         super(LinearDictSimple, self).__init__()

#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         self.num_keys = num_keys

#         self.linears = [nn.Linear(input_dim, output_dim, bias) for k in range(num_keys)]

#     def reset_parameters(self):
#         for l in self.linears:
#             l.reset_parameters()

#     def forward(self, X, keys):
#         out = torch.zeros((X.shape[0], self.output_dim), device=X.device)
        
#         keys = keys.squeeze(-1)
#         unique_keys = keys.unique()

#         for key in unique_keys:
#             idx = (keys == key)
#             X_key = X[key] 
            
#             # Lookup & reshape
#             # W_k: (N, input_dim, output_dim)
#             W = self.W.weight[key].view(self.input_dim, self.output_dim)

#             out[idx] = out[idx] + (X_key @ W)

#             if hasattr(self, 'bias'):
#                 b = self.bias.weight[key]
#                 out[idx] = out[idx] + b

#         return out

class LinearDictLora(nn.Module):
    def __init__(self, num_keys, input_dim, output_dim, bias=False, rank=50):
        super(LinearDictLora, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.num_keys = num_keys

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

        gain = torch.sqrt(torch.tensor(0.7))
        
        for k in range(self.num_keys):
            A = self.A.weight.data[k].view(self.input_dim, self.rank)
            nn.init.kaiming_uniform_(A, a=0, mode='fan_in', nonlinearity='linear')
            A.mul_(gain)
            
            B = self.B.weight.data[k].view(self.rank, self.output_dim)
            nn.init.kaiming_uniform_(B, a=0, mode='fan_in', nonlinearity='linear')
            B.mul_(gain)
        
        if hasattr(self, 'bias'):
            self.bias.reset_parameters()
            nn.init.zeros_(self.bias.weight)
    
    # LoRA implementation to handle scale of parameters
    def forward(self, X, keys):
        """
        X    : (N, input_dim)
        keys : (N,) or (N, 1)
        """
        # Base output: (N, output_dim)   
        out = self.W0(X)
        
        keys = keys.squeeze(-1)
        unique_keys = keys.unique()

        for key in unique_keys:
            idx = (keys == key)
            X_key = X[key] 
            
            # Lookup & reshape
            # A_k: (N, input_dim, rank)
            # B_k: (N, rank, output_dim)
            A = self.A.weight[key].view(self.input_dim, self.rank)
            B = self.B.weight[key].view(self.rank, self.output_dim)

            xA = X_key @ A
            delta_key = xA @ B

            out[idx] = out[idx] + delta_key

            if hasattr(self, 'bias'):
                b = self.bias.weight[key]
                out[idx] = out[idx] + b

        return out
            
        # # Lookup & reshape
        # # A_k: (N, input_dim, rank)
        # # B_k: (N, rank, output_dim)
        # A = self.A(keys).view(-1, self.input_dim, self.rank)
        # B = self.B(keys).view(-1, self.rank, self.output_dim)

        # # LoRA path using einsum
        # # X @ A_k → (N, rank)
        # z = torch.einsum("nd,ndr->nr", X, A)

        # # z @ B_k → (N, output_dim)
        # z = torch.einsum("nr,nro->no", z, B)

        # out = out + z

        # if hasattr(self, 'bias'):
        #     out = out + self.bias(keys)

        # return out

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
        