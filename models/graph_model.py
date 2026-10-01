import math

import torch
from torch_geometric.nn import GCNConv, TAGConv, ChebConv, GATConv, JumpingKnowledge
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, get_laplacian
from scipy.special import comb

class BernProp(MessagePassing):
    def __init__(self, K, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = torch.softmax(self.temp, dim=0)
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization='sym',
            dtype=x.dtype, num_nodes=x.size(self.node_dim)
        )
        edge_index2, norm2 = add_self_loops(
            edge_index1, -norm1, fill_value=2.0, num_nodes=x.size(self.node_dim)
        )
        tmp = [x]
        for _ in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            for _ in range(i + 1):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return f'{self.__class__.__name__}(K={self.K}, temp={self.temp})'


class BernConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=2, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.prop = BernProp(K)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.lin(x)
        x = self.prop(x, edge_index, edge_weight)
        return x






















    


















    


















        
















    
class GraphModel(nn.Module):
    def __init__(self, num_features, hidden_dim=256, output_dim=128, dropout=0.3, K=2):
        super().__init__()
        self.GCN1 = BernConv(num_features, hidden_dim, K=K)
        self.GCN2 = BernConv(hidden_dim, hidden_dim, K=K)
        self.GCN3 = BernConv(hidden_dim, output_dim, K=K)
        self.dropout = nn.Dropout(dropout)
        self.LP = nn.Linear(num_features, output_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        res_x = self.LP(x)

        x1 = self.GCN1(x, edge_index)
        x1 = self.ln1(x1)
        x1 = F.silu(x1)
        x1 = self.dropout(x1)

        x2 = self.GCN2(x1, edge_index)
        x2 = self.ln2(x2)
        x2 = F.silu(x2)
        x2 = self.dropout(x2)
        x2 = x1 + x2

        x3 = self.GCN3(x2, edge_index)
        x3 = self.ln3(x3)
        x3 = F.silu(x3 + res_x)
        
        return x3

















        

















class BernGraphModel(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_dim=256,
                 output_dim=128,
                 K=5,
                 dropout=0.3):
        super().__init__()

        self.conv1 = BernConv(num_features, hidden_dim, K=K)
        self.conv2 = BernConv(hidden_dim, hidden_dim, K=K)
        self.conv3 = BernConv(hidden_dim, output_dim, K=K)

        self.LP = nn.Linear(num_features, hidden_dim, bias=False)

        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index


        res_x = self.LP(x)

        x = self.norm1(x)
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = F.silu(x)

        x = self.norm2(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = res_x + x
        x = F.silu(x)

        x = self.norm3(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return F.silu(x)
class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_dim=256, output_dim=128, dropout=0.3):
        super().__init__()
        self.GCN1 = ChebConv(num_features, hidden_dim, K=2)
        self.GCN2 = ChebConv(hidden_dim, hidden_dim, K=2)
        self.GCN3 = ChebConv(hidden_dim, output_dim, K=2)
        self.dropout = nn.Dropout(dropout)
        self.LP = nn.Linear(num_features, hidden_dim)
        self.ln1 = nn.LayerNorm(num_features, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ln3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index


        res_x = self.LP(x)

        x = self.ln1(x)
        x = self.dropout(x)
        x = self.GCN1(x, edge_index)
        x = F.silu(x)

        x = self.ln2(x)
        x = self.dropout(x)
        x = self.GCN2(x, edge_index)
        x = res_x + x
        x = F.silu(x)

        x = self.GCN3(x, edge_index)
        return F.silu(x)