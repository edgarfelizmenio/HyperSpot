import torch

import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class IntraMA(nn.Module):
    # IntraMA - hyperedges are queriesw
    def __init__(self, input_dim, embedding_dim, num_heads, dropout):
        super(IntraMA, self).__init__()
        self.num_heads = num_heads
        
        self.norm_hyperedge = nn.LayerNorm(input_dim)
        self.norm_node = nn.LayerNorm(input_dim)
        
        self.hyperedge_query_linear = nn.Linear(input_dim, embedding_dim * num_heads, bias=False)
        self.node_key_linear = nn.Linear(input_dim, embedding_dim * num_heads, bias=False)
        self.node_value_linear = nn.Linear(input_dim, embedding_dim * num_heads, bias=False)

        self.scale = embedding_dim ** 0.5
        self.to_out = nn.Linear(embedding_dim * num_heads + input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
    
    def forward(self, hyperedge_embeddings, node_embeddings, incidence_matrix):
        hyperedge_embeddings = self.norm_hyperedge(hyperedge_embeddings)
        node_embeddings = self.norm_node(node_embeddings)

        q = self.hyperedge_query_linear(hyperedge_embeddings)
        k = self.node_key_linear(node_embeddings) 
        v = self.node_value_linear(node_embeddings)

        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        
        q = rearrange(q, 'n (h d) -> h n d', h = self.num_heads)
        k = rearrange(k, 'n (h d) -> h n d', h = self.num_heads)
        v = rearrange(v, 'n (h d) -> h n d', h = self.num_heads)

        # print(q.shape)
        # print(k.shape)
        # print(v.shape)

        # print(incidence_matrix.shape)
        incidence_matrix = repeat(incidence_matrix, 'i j -> h j i', h=self.num_heads)

        attn_scores = torch.einsum('hid,hjd->hij', q, k) / self.scale
        weights = F.softmax(attn_scores, dim=-1)
        weights = self.dropout(weights)

        output = torch.einsum('hij,hjd->hid', weights, v)
        output = rearrange(output, 'h n d ->n (h d)', h = self.num_heads)
        output = self.act(output)
        # output = output[:spot_expr.shape[0]] # discard cell type hyperedge embeddings
        
        final_output = torch.cat((hyperedge_embeddings, output), dim=-1)
        final_output = self.to_out(final_output)

        return final_output

class InterMA(nn.Module):
    # InterMA - hyperedges are keys and values
    def __init__(self, input_dim, embedding_dim, num_heads, dropout):
        super(InterMA, self).__init__()
        self.num_heads = num_heads
        
        self.norm_node = nn.LayerNorm(input_dim)
        self.norm_hyperedge = nn.LayerNorm(input_dim)

        self.node_query_linear = nn.Linear(input_dim, embedding_dim * num_heads, bias=False)
        self.hyperedge_key_linear = nn.Linear(input_dim, embedding_dim * num_heads, bias=False)
        self.hyperedge_value_linear = nn.Linear(input_dim, embedding_dim * num_heads, bias=False)
        
        self.scale = embedding_dim ** 0.5
        self.to_out = nn.Linear(embedding_dim * num_heads + input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
    
    def forward(self, node_embeddings, hyperedge_embeddings, incidence_matrix):
        node_embeddings = self.norm_node(node_embeddings)
        hyperedge_embeddings = self.norm_hyperedge(hyperedge_embeddings)

        q = self.node_query_linear(node_embeddings)        
        k = self.hyperedge_key_linear(hyperedge_embeddings) 
        v = self.hyperedge_value_linear(hyperedge_embeddings)

        q = rearrange(q, 'n (h d) -> h n d', h = self.num_heads)
        k = rearrange(k, 'n (h d) -> h n d', h = self.num_heads)
        v = rearrange(v, 'n (h d) -> h n d', h = self.num_heads)
        incidence_matrix = repeat(incidence_matrix, 'i j -> h i j', h=self.num_heads)

        # print(q.shape, k.shape)
        attn_scores = torch.einsum('hid,hjd->hij', q, k) / self.scale
        weights = F.softmax(attn_scores, dim=-1)
        weights = self.dropout(weights)

        output = torch.einsum('hij,hjd->hid', weights, v)
        output = rearrange(output, 'h n d -> n (h d)', h = self.num_heads)
        output = self.act(output)

        final_output = torch.cat((node_embeddings, output), dim=-1)
        final_output = self.to_out(final_output)

        return final_output

class MutualAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, dropout):
        super(MutualAttention, self).__init__()
        
        self.intraMA = IntraMA(input_dim, embedding_dim, num_heads, dropout)
        self.interMA = InterMA(input_dim, embedding_dim, num_heads, dropout)

    def forward(self, node_embeddings, hyperedge_embeddings, incidence_matrix):
        hyperedge_embeddings = self.intraMA(hyperedge_embeddings, node_embeddings, incidence_matrix)
        node_embeddings = self.interMA(node_embeddings, hyperedge_embeddings, incidence_matrix)
        
        return node_embeddings, hyperedge_embeddings

class STHypergraphTransformerBlock(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, dropout):
        super(STHypergraphTransformerBlock, self).__init__()
        
        self.MA = MutualAttention(input_dim, embedding_dim, num_heads, dropout)
    
    def forward(self, node_input_embeddings, hyperedge_input_embeddings, incidence_matrix):
        
        node_embeddings, hyperedge_embeddings = self.MA(node_input_embeddings, 
                                                        hyperedge_input_embeddings,
                                                        incidence_matrix)
        
        # skip connections
        node_embeddings = node_embeddings + node_input_embeddings
        hyperedge_embeddings = hyperedge_embeddings + hyperedge_input_embeddings
        return node_embeddings, hyperedge_embeddings     

class STHypergraphTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_hyperedges, num_heads, num_layers=1, dropout=0.1, hyperedge_feats="pool"):
        super(STHypergraphTransformer, self).__init__()
        self.hyperedge_feats = hyperedge_feats
        if self.hyperedge_feats == "learned":
            print("Hyperedge Features: Learned")
            self.hyperedge_embedding = nn.Embedding(num_hyperedges, input_dim)
        else:
            print("Hyperedge Features: Pooled")
        
        self.layers = nn.ModuleList([STHypergraphTransformerBlock(input_dim, embedding_dim, num_heads, dropout) for i in range(num_layers)])

    def __pool_embedding(self, node_embeddings, incidence_matrix):
        hyperedge_embeddings = (incidence_matrix.t() @ node_embeddings) / incidence_matrix.sum(dim=0).unsqueeze(dim=1)
        return hyperedge_embeddings
    
    def forward(self, node_feats, hyperedge_ids, incidence_matrix):
        node_embeddings = node_feats
        if self.hyperedge_feats == "learned":
            hyperedge_embeddings = self.hyperedge_embedding(hyperedge_ids)
        else:
            hyperedge_embeddings = self.__pool_embedding(node_embeddings, incidence_matrix)
        
        for layer in self.layers:
            node_embeddings, hyperedge_embeddings = layer(node_embeddings, hyperedge_embeddings, incidence_matrix)            

        return node_embeddings, hyperedge_embeddings 
    
# class STHypergraphTransformer(nn.Module):
#     def __init__(self, input_dim, num_cell_types, num_spots, embedding_dim, num_heads, num_layers=1, dropout=0.1):
#         super(STHypergraphTransformer, self).__init__()
#         self.spot_embedding = SpotEmbedding(num_spots, embedding_dim)
#         self.cell_type_embedding = CellTypeEmbedding(num_cell_types, embedding_dim)
        
#         self.layers = nn.ModuleList([STHypergraphTransformerBlock(input_dim, 
#                                                                   embedding_dim, 
#                                                                   num_heads, 
#                                                                   dropout) for i in range(num_layers)])
#         print(self)
        
#     def forward(self, cell_expr, hyperedge_ids, incidence_matrix):
#         spot_embedding = self.spot_embedding(hyperedge_ids)
#         cell_type_embedding = self.cell_type_embedding()

        
#         hyperedge_embeddings = torch.cat([spot_embedding, cell_type_embedding], dim=0)
#         cell_embeddings = cell_expr


#         for l in range(len(self.layers)):
#             cell_embeddings, hyperedge_embeddings, intraMA_score, interMA_score = self.layers[l](cell_embeddings, 
#                                                                                           hyperedge_embeddings,
#                                                                                           incidence_matrix)

#         hyperedge_embeddings = hyperedge_embeddings[:hyperedge_ids.shape[0]]
        
#         # skip connections
#         return cell_embeddings, hyperedge_embeddings, intraMA_score, interMA_score        


# class STHypergraphTransformerBlock(nn.Module):
#     def __init__(self, input_dim, embedding_dim, num_heads, dropout):
#         super(STHypergraphTransformerBlock, self).__init__()
#         self.MA = MutualAttention(input_dim, embedding_dim, num_heads, dropout)

#     def forward(self, cell_expr, hyperedge_embeddings, incidence_matrix):
#         cell_embeddings, hyperedge_embeddings, intraMA_score, interMA_score = self.MA(cell_expr, 
#                                                                                       hyperedge_embeddings,
#                                                                                       incidence_matrix)

#         # hyperedge_embeddings = hyperedge_embeddings[:hyperedge_ids.shape[0]]
        
#         # skip connections
#         cell_embeddings += cell_expr
#         hyperedge_embeddings += spot_embedding
#         return cell_embeddings, hyperedge_embeddings, intraMA_score, interMA_score      
        
# loss
# 1. cell contrastive loss -> cells of the same type must be closer to each other 
# 2. spatial loss -> adjacent cells must be more similar to each other
# 3. spot contrastive loss -> adjacent spots must be closer to each other
