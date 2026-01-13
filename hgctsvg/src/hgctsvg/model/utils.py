import torch
import torch.nn as nn
                
def compute_fc_layer_sizes(in_feats, out_feats, num_layers=5):
    return torch.linspace(in_feats, out_feats, num_layers).round().int()

def create_fc(layer_sizes, bias, Activation=nn.PReLU, dropout=None):
    layers = nn.Sequential()
    for i in range(1, len(layer_sizes)):
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i], bias))
        layers.append(nn.BatchNorm1d(layer_sizes[i]))
        layers.append(Activation(layer_sizes[i]))
    return layers