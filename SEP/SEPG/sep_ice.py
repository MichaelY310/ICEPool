#!/usr/bin/env python
# encoding: utf-8
import torch
import numpy as np
import torch.nn as nn
import torch_geometric.nn.models
from torch import Tensor
from typing import Callable
from functools import reduce
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_add_pool as gsp
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from typing import Optional
from torch.nn import Parameter
from torch_geometric.utils import softmax

class SEPooling(MessagePassing):
    def __init__(self, nn: Callable, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None, **kwargs) -> Tensor:
        # out = self.propagate(edge_index, x=x, size=size)
        out = self.propagate(edge_index, x=x, size=size[[1,0]])
        return out
        # return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class SVDPooling(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super().__init__(**kwargs)

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj, weight: Tensor, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, weight=weight, size=size[[1,0]])
        return out

    def message(self, x_j, weight):
        return torch.cat([weight.view(-1, 1) * x_j, x_j], dim=-1)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
















class EGAT(MessagePassing):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = True,
        args = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.args = args
        self.dropout = self.args.egat_dropout
        self.alpha = self.args.egat_alpha

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.W = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        self.a1 = Linear(out_channels, 1, bias=False,
                          weight_initializer='glorot')
        self.a2 = Linear(out_channels, 1, bias=False,
                          weight_initializer='glorot')
        self.W_out = Linear(out_channels * 3, out_channels, bias=False,
                          weight_initializer='glorot')

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.W.reset_parameters()
        zeros(self.bias)

    def forward(self, h, edge_index, edgefeat) -> Tensor:

        Wh = self.W(h)
        Wh1 = self.a1(Wh)[edge_index[0]]
        Wh2 = self.a2(Wh)[edge_index[1]]

        attention = self.edge_updater(edge_index, alpha=(Wh1, Wh2), edge_attr=edgefeat)
        out = self.propagate(edge_index, x=Wh, edge_weight=attention)

        return out

    def message(self, x_j, edge_weight) -> Tensor:
        # Choice 1: simply concat embeddings corresponding to 3 edgefeats
        # Choice 2: take average
        # Choice 3: concat and then pass through a linear layer
        out = edge_weight.unsqueeze(2) * x_j.unsqueeze(1)
        out = (out).reshape(out.shape[0], -1)
        return self.W_out(out)

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, dim_size=None) -> Tensor:
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.alpha)
        alpha = alpha * edge_attr
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha











class SEP_ICE(torch.nn.Module):
    def __init__(self, args):
        super(SEP_ICE, self).__init__()
        self.args = args
        self.num_features = args.num_features  # input_dim
        self.nhid = args.hidden_dim  # hidden dim
        self.num_classes = args.num_classes  # output dim
        self.dropout_ratio = args.final_dropout
        self.convs = self.get_convs()
        self.sepools = self.get_sepool()
        self.global_pool = gsp if args.global_pooling == 'sum' else gap
        self.classifier = self.get_classifier()

    def __process_layer_edgeIndex(self, batch_data, layer=0):
        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_data):
            start_idx.append(start_idx[i] + graph['node_size'][layer])
            edge_mat_list.append(graph['graph_mats'][layer] + start_idx[i])
        edge_index = torch.cat(edge_mat_list, 1)
        return edge_index.to(self.args.device)

    def __process_sep_edgeIndex(self, batch_data, layer=1):
        edge_mat_list = []
        start_pdx = [0]
        start_idx = [0]
        for i, graph in enumerate(batch_data):
            start_pdx.append(start_pdx[i] + graph['node_size'][layer-1])
            start_idx.append(start_idx[i] + graph['node_size'][layer])
            edge_mat_list.append(torch.LongTensor(graph['edges'][layer])
                                 + torch.LongTensor([start_idx[i], start_pdx[i]]))
        edge_index = torch.cat(edge_mat_list, 0).T
        return edge_index.to(self.args.device)

    def __process_sep_size(self, batch_data, layer=1):
        size = [(graph['node_size'][layer-1], graph['node_size'][layer]) for graph in batch_data]
        return np.array(size).sum(axis=0)

    def __process_batch(self, batch_data, layer=0):
        batch = [[i] * graph['node_size'][layer] for i, graph in enumerate(batch_data)]
        batch = reduce(lambda x, y: x+y, batch)
        return torch.tensor(batch, dtype=torch.long).to(self.args.device)

    def get_convs(self):
        convs = nn.ModuleList()
        _input_dim = self.num_features
        _output_dim = self.nhid
        if self.args.svdpool:
            _output_dim = _output_dim // 2
        recovered = False
        for _ in range(self.args.num_convs):
            if (not recovered) and (self.args.svdpool) and (_ >= self.args.num_svd_pools or _ == self.args.num_convs - 1):  # The last layer has no svdpool, so don't half the output_dim. Also don't half the output_dim if we are no longer using svd pool
                _output_dim = _output_dim * 2
                recovered = True

            if self.args.conv == 'GCN':
                conv = GCNConv(_input_dim, _output_dim)
            elif self.args.conv == 'GAT':
                conv = GATConv(_input_dim, _output_dim, self.args.num_head, concat=False)
            elif self.args.conv == 'Cheb':
                conv = ChebConv(_input_dim, _output_dim, K=2)
            elif self.args.conv == 'SAGE':
                conv = SAGEConv(_input_dim, _output_dim)
            elif self.args.conv == 'GAT2':
                conv = GATv2Conv(_input_dim, _output_dim, self.args.num_head, concat=False)
            elif self.args.conv == 'Transformer':
                conv = TransformerConv(_input_dim, _output_dim, self.args.num_head, concat=False)
            elif self.args.conv == 'GIN':
                conv = GINConv(
                    nn.Sequential(
                        nn.Linear(_input_dim, _output_dim),
                        nn.ReLU(),
                        nn.Linear(_output_dim, _output_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(_output_dim),
                    ), train_eps=False)
            elif self.args.conv == "EGAT":
                if _ == 0 or _ >= self.args.num_ce_aggs + 1:
                    conv = GCNConv(_input_dim, _output_dim)
                else:
                    conv = EGAT(_input_dim, _output_dim, args=self.args)
            elif self.args.conv == "Edgefeat-GAT":
                if _ == 0 or _ >= self.args.num_ce_aggs + 1:
                    conv = GCNConv(_input_dim, _output_dim)
                else:
                    conv = GATConv(_input_dim, _output_dim, edge_dim=3, dropout=self.args.egat_dropout, negative_slope=self.args.egat_alpha)
            convs.append(conv)
            _input_dim = _output_dim
            if self.args.svdpool and _ < self.args.num_svd_pools: # If svdpool is previously used, double the input_dim to match the concatenated tensor's dim.
                _input_dim = _input_dim * 2
        return convs

    def get_sepool(self):
        pools = nn.ModuleList()
        _input_dim = self.nhid
        _output_dim = self.nhid
        for _ in range(self.args.tree_depth-1):
            if self.args.svdpool:
                if _ < self.args.num_svd_pools:
                    pool = SVDPooling()
                else:
                    pool = SEPooling(
                        nn.Sequential(
                            nn.Linear(_input_dim, _output_dim),
                            nn.ReLU(),
                            nn.Linear(_output_dim, _output_dim),
                            nn.ReLU(),
                            nn.BatchNorm1d(_output_dim),
                        ))
            else:
                pool = SEPooling(
                    nn.Sequential(
                        nn.Linear(_input_dim, _output_dim),
                        nn.ReLU(),
                        nn.Linear(_output_dim, _output_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(_output_dim),
                    ))
            pools.append(pool)
            _input_dim = _output_dim
        return pools

    def get_classifier(self):
        init_dim = self.nhid * self.args.num_convs
        if self.args.link_input:
            init_dim += self.num_features
        return nn.Sequential(
            nn.Linear(init_dim, self.nhid),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.num_classes)
        )

    def forward(self, batch_data, **kwargs):
        x = xIn = torch.cat([t['node_features'] for t in batch_data], dim=0).to(self.args.device)
        xs = []
        for _ in range(self.args.num_convs):
            # mp
            edge_index = self.__process_layer_edgeIndex(batch_data, _)
            if self.args.conv == "EGAT":
                if _ == 0 or _ >= self.args.num_ce_aggs + 1:
                    x = F.relu(self.convs[_](x, edge_index))
                else:
                    edgefeat = torch.cat([t['edgefeats'][_-1] for t in batch_data], dim=0).to(self.args.device, dtype=torch.float32)
                    x = F.relu(self.convs[_](x, edge_index, edgefeat=edgefeat))
            elif self.args.conv == "Edgefeat-GAT":
                if _ == 0 or _ >= self.args.num_ce_aggs + 1:
                    x = F.relu(self.convs[_](x, edge_index))
                else:
                    edgefeat = torch.cat([t['edgefeats'][_-1] for t in batch_data], dim=0).to(self.args.device, dtype=torch.float32)
                    x = F.relu(self.convs[_](x, edge_index, edge_attr=edgefeat))
            else:
                x = F.relu(self.convs[_](x, edge_index))
            # sep
            if _ < self.args.tree_depth - 1:
                edge_index = self.__process_sep_edgeIndex(batch_data, _+1)
                size = self.__process_sep_size(batch_data, _+1)
                edge_weight = torch.cat([t['edgepool_weights'][_] for t in batch_data], dim=0).to(self.args.device, dtype=torch.float32)
                if self.args.svdpool and _ < self.args.num_svd_pools:
                    x = F.relu(self.sepools[_](x, edge_index, size=size, weight=edge_weight))
                else:
                    x = F.relu(self.sepools[_](x, edge_index, size=size))
            xs.append(x)

        pooled_xs = []
        if self.args.link_input:
            batch = self.__process_batch(batch_data, 0)
            pooled_x = self.global_pool(xIn, batch)
            pooled_xs.append(pooled_x)
        for _, x in enumerate(xs):
            batch = self.__process_batch(batch_data, min(_+1, self.args.tree_depth-1))
            pooled_x = self.global_pool(x, batch)
            pooled_xs.append(pooled_x)

        # For jumping knowledge scheme
        x = torch.cat(pooled_xs, dim=1)
        # For Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)
