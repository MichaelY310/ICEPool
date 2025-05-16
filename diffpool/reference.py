import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from EGAT import *

import numpy as np

from set2set import Set2Set


class Pool_Edges(nn.Module):
    def __init__(self, edge_pool_matrices, edge_pool_weight, X_concat_Singular, device):
        super(Pool_Edges, self).__init__()
        self.edge_pool_matrices = edge_pool_matrices
        self.device = device
        self.edge_pool_weight = edge_pool_weight
        self.X_concat_Singular = X_concat_Singular

    def forward(self, x, coarsened_x):
        edge_pool_matrix = self.edge_pool_matrices

        edge_pool_matrix = edge_pool_matrix.type(torch.FloatTensor).to(self.device)
        edge_pool_matrix = torch.transpose(edge_pool_matrix, 1, 2)
        edgepool_result = torch.matmul(edge_pool_matrix, x)

        if self.X_concat_Singular:  # Concatenate coarsened_x and edge pooling
            x_pooled = torch.cat((coarsened_x, edgepool_result), dim=-1)
        else:  # Add coarsened_x and edge pooling
            x_pooled = coarsened_x + edgepool_result

        return x_pooled


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True, device="cuda"):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.device = device
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).to(device))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).to(device))
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            # print(y[0][0])
        return y


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, device="cuda", args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1
        self.device = device

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias, device=self.device)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias, device=self.device)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias, device=self.device)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).to(self.device)

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).to(self.device)
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        # print(output.size())
        return ypred

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().to(self.device)
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        # return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class GcnSet2SetEncoder(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnSet2SetEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                num_layers, pred_hidden_dims, concat, bn, dropout, args=args)
        self.s2s = Set2Set(self.pred_input_dim, self.pred_input_dim * 2)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out = self.s2s(embedding_tensor)
        # out, _ = torch.max(embedding_tensor, dim=1)
        ypred = self.pred_model(out)
        return ypred


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, device="cuda", args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                    num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat,
                                                    args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.device = device

        # GC
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling + 1), pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        # self.assign_tensor = self.gcn_forward(x_a, adj,
        #        self.assign_conv_first_modules[0], self.assign_conv_block_modules[0], self.assign_conv_last_modules[0],
        #        embedding_mask)
        ## [batch_size x num_nodes x next_lvl_num_nodes]
        # self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
        # if embedding_mask is not None:
        #    self.assign_tensor = self.assign_tensor * embedding_mask
        # [batch_size x num_nodes x embedding_dim]
        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)

        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(x_a, adj,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                                                  self.assign_conv_last_modules[i],
                                                  embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(self.device))
            # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1 - adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss


class DET_SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, max_num_clusters_fordet, input_dim, hidden_dim, embedding_dim, label_dim,
                 num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, device="cuda", args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(DET_SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                        num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat,
                                                        args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.max_num_nodes = max_num_nodes
        self.max_num_clusters_fordet = max_num_clusters_fordet
        self.device = device

        # GC
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling + 1), pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, det_assign_matrix, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        # self.assign_tensor = self.gcn_forward(x_a, adj,
        #        self.assign_conv_first_modules[0], self.assign_conv_block_modules[0], self.assign_conv_last_modules[0],
        #        embedding_mask)
        ## [batch_size x num_nodes x next_lvl_num_nodes]
        # self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
        # if embedding_mask is not None:
        #    self.assign_tensor = self.assign_tensor * embedding_mask
        # [batch_size x num_nodes x embedding_dim]
        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)

        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            self.assign_tensor = det_assign_matrix

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(DET_SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(self.device))
            # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1 - adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss


class ICE_SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, use_simple_gat=False, egat_hidden_dims=[128, 128], egat_dropout=0.6,
                 egat_alpha=0.2, egat_num_heads=[1, 3], pooling_size_real=8, DSN=False, device="cuda", args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(ICE_SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                        num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat,
                                                        args=args)
        self.device = device

        print("USING ICE METHOD")
        add_self = not concat
        self.max_num_nodes = max_num_nodes
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.num_layers = num_layers
        self.pred_hidden_dims = pred_hidden_dims
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # # GC
        # self.conv_first_after_pool = nn.ModuleList()
        # self.conv_block_after_pool = nn.ModuleList()
        # self.conv_last_after_pool = nn.ModuleList()
        # for i in range(num_pooling):
        #     # use self to register the modules in self.modules()
        #     conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
        #         self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
        #         add_self, normalize=True, dropout=dropout)
        #     self.conv_first_after_pool.append(conv_first2)
        #     self.conv_block_after_pool.append(conv_block2)
        #     self.conv_last_after_pool.append(conv_last2)

        # assignment

        self.pooling_size_real = pooling_size_real

        self.assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_num_layers = assign_num_layers
        self.assign_input_dim = assign_input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes / self.pooling_size_real)
        for i in range(num_pooling):
            self.assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim / self.pooling_size_real)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        # EGAT
        self.DSN = DSN
        self.edgefeat_dim = 3
        self.use_simple_gat = use_simple_gat
        self.pred_input_dim = self.edgefeat_dim * embedding_dim
        self.egat_hidden_dims = egat_hidden_dims
        self.egat_dropout = egat_dropout
        self.egat_alpha = egat_alpha
        self.egat_num_heads = egat_num_heads

        print("Using EGAT edge feat")
        self.egat_list = nn.ModuleList()
        for i in range(num_pooling):
            if i == 0:
                # The first layer is still GNN, so we need to use the old pred_input_dim
                if use_simple_gat:
                    self.egat_list.append(
                        BatchGraphAttentionLayer((self.hidden_dim * (num_layers - 1) + embedding_dim) * 1,
                                                 embedding_dim, self.egat_dropout, self.egat_alpha).to(
                            device))
                else:
                    self.egat_list.append(
                        BatchGATFeatureExtractor((self.hidden_dim * (num_layers - 1) + embedding_dim) * 1,
                                                 self.edgefeat_dim, self.egat_hidden_dims, embedding_dim,
                                                 self.egat_dropout, self.egat_alpha, self.egat_num_heads).to(
                            device))
            else:
                if use_simple_gat:
                    self.egat_list.append(
                        BatchGraphAttentionLayer(self.pred_input_dim * 1, embedding_dim, self.egat_dropout,
                                                 self.egat_alpha).to(
                            device))
                else:
                    self.egat_list.append(
                        BatchGATFeatureExtractor(self.pred_input_dim * 1, self.edgefeat_dim, self.egat_hidden_dims,
                                                 embedding_dim, self.egat_dropout, self.egat_alpha,
                                                 self.egat_num_heads).to(
                            device))

        self.pred_model = self.build_pred_layers(
            self.hidden_dim * (self.num_layers - 1) + self.embedding_dim +
            self.pred_input_dim * self.num_pooling,
            self.pred_hidden_dims,
            self.label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        # self.assign_tensor = self.gcn_forward(x_a, adj,
        #        self.assign_conv_first_modules[0], self.assign_conv_block_modules[0], self.assign_conv_last_modules[0],
        #        embedding_mask)
        ## [batch_size x num_nodes x next_lvl_num_nodes]
        # self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
        # if embedding_mask is not None:
        #    self.assign_tensor = self.assign_tensor * embedding_mask
        # [batch_size x num_nodes x embedding_dim]
        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        # Output shape: (num_layers-1) * hidden_size + embedding_size

        out, _ = torch.max(embedding_tensor,
                           dim=1)  # Sum over nodes: batch_size * [(num_layers-1) * hidden_size + embedding_size]
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            # adj is the current A. To get entropy we need the previous adj
            # Calculate the assign matrix using the original x instead of the embedding tensor
            self.assign_tensor = self.gcn_forward(x_a, adj,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                                                  self.assign_conv_last_modules[i],
                                                  embedding_mask)  # The embdding mask restricts the node number
            # DiffPool doesn't reduce the size of the graph at all and even increases it because the number of clusters
            # after pooling is usually set to a fixed large number. For example, the mean number of nodes in Enzymes
            # dataset is about 30, but the code set max_num_nodes to 1000, which means with 0.1 pooling ratio, the
            # coarsened graph would have 1000*0.1=100 nodes, which is much larger than the average size.

            # As a result we use assign_mask to REALLY restrict the coarsened graph size.
            # assign_mask = self.construct_mask(self.assign_tensor.shape[-1], batch_num_nodes // self.pooling_size_real)
            # self.assign_tensor = self.assign_tensor * assign_mask.transpose(-1, -2)

            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            assign_mask = self.construct_mask(self.assign_tensor.shape[-1], batch_num_nodes // self.pooling_size_real)
            self.assign_tensor = self.assign_tensor * embedding_mask * assign_mask.transpose(-1, -2)
            self.assign_prob = torch.nn.Softmax(2)(
                self.assign_tensor * max_num_nodes) * embedding_mask * assign_mask.transpose(-1, -2)

            H = self.batch_connection_entropy(adj, self.assign_prob)

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            # Concatenate H, H.T, A to get edge_feat
            edgefeat = torch.cat([adj.unsqueeze(-1), H.unsqueeze(-1), H.transpose(-1, -2).unsqueeze(-1)], dim=-1)
            if self.DSN:
                edgefeat = batch_DSN_fast(edgefeat)
            edgefeat_mask = (adj != 0).float()
            embedding_tensor = self.egat_forward(self.egat_list[i], x, edgefeat, assign_mask,
                                                 edgefeat_mask)

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def egat_forward(self, egat, embedding_tensor, edgefeat, embedding_mask, edgefeat_mask):
        edgefeat = torch.permute(edgefeat, (0, 3, 1, 2))
        if self.use_simple_gat:
            x_tensor, _ = egat(embedding_tensor, edgefeat, edgefeat_mask)
        else:
            x_tensor = egat(embedding_tensor, edgefeat, edgefeat_mask)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask

        return x_tensor

    def connection_entropy(self, A, S):
        A_ext = A - (S @ S.T) * A
        # SA[i, j] is the number of edges from node i to cluster j
        SA = S.T @ A_ext

        STA_extS = S.T @ A_ext @ S
        S_A = SA.unsqueeze(1)
        S_A = S_A * S.T.unsqueeze(0)
        P = S_A / (STA_extS.unsqueeze(-1) + 1e-20)
        H = -(P * torch.log(P + 1e-20)).sum(-1)

        return H.T

    def batch_connection_entropy(self, A, S):
        A_ext = A - torch.bmm(S, S.transpose(-1, -2)) * A
        SA = torch.bmm(S.transpose(-1, -2), A_ext)

        STA_extS = torch.bmm(torch.bmm(S.transpose(-1, -2), A_ext), S)
        S_A = SA.unsqueeze(2)
        S_A = S_A * S.transpose(-1, -2).unsqueeze(1)
        P = S_A / (STA_extS.unsqueeze(-1) + 1e-20)
        H = -(P * torch.log(P + 1e-20)).sum(-1)

        return H.transpose(-1, -2)

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(ICE_SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(self.device))
            # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1 - adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss


class ICE_DET_SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, max_num_clusters_fordet, input_dim, hidden_dim, embedding_dim, label_dim,
                 num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, use_simple_gat=False, egat_hidden_dims=[128, 128], egat_dropout=0.6,
                 egat_alpha=0.2, egat_num_heads=[1, 3], pooling_size_real=8, DSN=False, device="cuda", args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(ICE_DET_SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                            num_layers, pred_hidden_dims=pred_hidden_dims,
                                                            concat=concat,
                                                            args=args)

        self.device = device

        print("USING ICE METHOD")
        add_self = not concat
        self.max_num_nodes = max_num_nodes
        self.max_num_clusters_fordet = max_num_clusters_fordet
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.num_layers = num_layers
        self.pred_hidden_dims = pred_hidden_dims
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # # GC
        # self.conv_first_after_pool = nn.ModuleList()
        # self.conv_block_after_pool = nn.ModuleList()
        # self.conv_last_after_pool = nn.ModuleList()
        # for i in range(num_pooling):
        #     # use self to register the modules in self.modules()
        #     conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
        #         self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
        #         add_self, normalize=True, dropout=dropout)
        #     self.conv_first_after_pool.append(conv_first2)
        #     self.conv_block_after_pool.append(conv_block2)
        #     self.conv_last_after_pool.append(conv_last2)

        # assignment

        self.pooling_size_real = pooling_size_real

        self.assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_num_layers = assign_num_layers
        self.assign_input_dim = assign_input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes / self.pooling_size_real)
        for i in range(num_pooling):
            self.assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim / self.pooling_size_real)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        # EGAT
        self.DSN = DSN
        self.edgefeat_dim = 3
        self.use_simple_gat = use_simple_gat
        self.pred_input_dim = self.edgefeat_dim * embedding_dim
        self.egat_hidden_dims = egat_hidden_dims
        self.egat_dropout = egat_dropout
        self.egat_alpha = egat_alpha
        self.egat_num_heads = egat_num_heads

        print("Using EGAT edge feat")
        self.egat_list = nn.ModuleList()
        for i in range(num_pooling):
            if i == 0:
                # The first layer is still GNN, so we need to use the old pred_input_dim
                if use_simple_gat:
                    self.egat_list.append(
                        BatchGraphAttentionLayer((self.hidden_dim * (num_layers - 1) + embedding_dim) * 1,
                                                 embedding_dim, self.egat_dropout, self.egat_alpha).to(
                            device))
                else:
                    self.egat_list.append(
                        BatchGATFeatureExtractor((self.hidden_dim * (num_layers - 1) + embedding_dim) * 1,
                                                 self.edgefeat_dim, self.egat_hidden_dims, embedding_dim,
                                                 self.egat_dropout, self.egat_alpha, self.egat_num_heads).to(
                            device))
            else:
                if use_simple_gat:
                    self.egat_list.append(
                        BatchGraphAttentionLayer(self.pred_input_dim * 1, embedding_dim, self.egat_dropout,
                                                 self.egat_alpha).to(
                            device))
                else:
                    self.egat_list.append(
                        BatchGATFeatureExtractor(self.pred_input_dim * 1, self.edgefeat_dim, self.egat_hidden_dims,
                                                 embedding_dim, self.egat_dropout, self.egat_alpha,
                                                 self.egat_num_heads).to(
                            device))

        self.pred_model = self.build_pred_layers(
            self.hidden_dim * (self.num_layers - 1) + self.embedding_dim +
            self.pred_input_dim * self.num_pooling,
            self.pred_hidden_dims,
            self.label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, det_assign_matrix, det_edgefeat_list, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        # self.assign_tensor = self.gcn_forward(x_a, adj,
        #        self.assign_conv_first_modules[0], self.assign_conv_block_modules[0], self.assign_conv_last_modules[0],
        #        embedding_mask)
        ## [batch_size x num_nodes x next_lvl_num_nodes]
        # self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
        # if embedding_mask is not None:
        #    self.assign_tensor = self.assign_tensor * embedding_mask
        # [batch_size x num_nodes x embedding_dim]
        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        # Output shape: (num_layers-1) * hidden_size + embedding_size

        out, _ = torch.max(embedding_tensor,
                           dim=1)  # Sum over nodes: batch_size * [(num_layers-1) * hidden_size + embedding_size]
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_prob = det_assign_matrix
            self.assign_tensor = self.assign_prob
            assign_mask = self.construct_mask(self.assign_prob.shape[-1], batch_num_nodes_after_pool)

            # Option 1: Calculate entropy during training
            # H = self.batch_connection_entropy(adj, self.assign_prob)
            # Option 2: Use Precalculated CE
            H = det_edgefeat_list

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            # Concatenate H, H.T, A to get edge_feat
            edgefeat = torch.cat([adj.unsqueeze(-1), H.unsqueeze(-1), H.transpose(-1, -2).unsqueeze(-1)], dim=-1)
            if self.DSN:
                edgefeat = batch_DSN_fast(edgefeat)
            edgefeat_mask = (adj != 0).float()
            embedding_tensor = self.egat_forward(self.egat_list[i], x, edgefeat, assign_mask,
                                                 edgefeat_mask)

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def egat_forward(self, egat, embedding_tensor, edgefeat, embedding_mask, edgefeat_mask):
        edgefeat = torch.permute(edgefeat, (0, 3, 1, 2))
        if self.use_simple_gat:
            x_tensor, _ = egat(embedding_tensor, edgefeat, edgefeat_mask)
        else:
            x_tensor = egat(embedding_tensor, edgefeat, edgefeat_mask)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask

        return x_tensor

    def connection_entropy(self, A, S):
        A_ext = A - (S @ S.T) * A
        # SA[i, j] is the number of edges from node i to cluster j
        SA = S.T @ A_ext

        STA_extS = S.T @ A_ext @ S
        S_A = SA.unsqueeze(1)
        S_A = S_A * S.T.unsqueeze(0)
        P = S_A / (STA_extS.unsqueeze(-1) + 1e-20)
        H = -(P * torch.log(P + 1e-20)).sum(-1)

        return H.T

    def batch_connection_entropy(self, A, S):
        A_ext = A - torch.bmm(S, S.transpose(-1, -2)) * A
        SA = torch.bmm(S.transpose(-1, -2), A_ext)

        STA_extS = torch.bmm(torch.bmm(S.transpose(-1, -2), A_ext), S)
        S_A = SA.unsqueeze(2)
        S_A = S_A * S.transpose(-1, -2).unsqueeze(1)
        P = S_A / (STA_extS.unsqueeze(-1) + 1e-20)
        H = -(P * torch.log(P + 1e-20)).sum(-1)

        return H.transpose(-1, -2)

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(ICE_DET_SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(self.device))
            # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1 - adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss


class SVD_DET_SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, max_num_clusters_fordet, input_dim, hidden_dim, embedding_dim, label_dim,
                 num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1,

                 edge_pool_weight=0.5,
                 X_concat_Singular=True,

                 device="cuda",
                 args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SVD_DET_SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                            num_layers, pred_hidden_dims=pred_hidden_dims,
                                                            concat=concat,
                                                            args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.max_num_nodes = max_num_nodes
        self.max_num_clusters_fordet = max_num_clusters_fordet
        self.device = device
        self.edge_pool_weight = edge_pool_weight
        self.X_concat_Singular = X_concat_Singular

        # GC
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            mul = 2 if self.X_concat_Singular else 1
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                self.pred_input_dim * mul, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling + 1), pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, det_assign_matrix, det_edgepool_matrices_dic, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            pool = Pool_Edges(det_edgepool_matrices_dic,
                              edge_pool_weight=self.edge_pool_weight,
                              X_concat_Singular=self.X_concat_Singular,
                              device=self.device)
            self.assign_tensor = det_assign_matrix
            coarsened_x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            x = pool(embedding_tensor, coarsened_x)

            # update pooled features and adj matrix
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(SVD_DET_SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(self.device))
            # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1 - adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss


class ICE_SVD_DET_SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, max_num_clusters_fordet, input_dim, hidden_dim, embedding_dim, label_dim,
                 num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1,

                 use_simple_gat=False, egat_hidden_dims=[128, 128], egat_dropout=0.6, egat_alpha=0.2,
                 egat_num_heads=[1, 3], pooling_size_real=8, DSN=False,

                 edge_pool_weight=0.5,
                 X_concat_Singular=True,

                 device="cuda",
                 args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(ICE_SVD_DET_SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                                num_layers, pred_hidden_dims=pred_hidden_dims,
                                                                concat=concat,
                                                                args=args)
        add_self = not concat
        self.max_num_nodes = max_num_nodes
        self.max_num_clusters_fordet = max_num_clusters_fordet
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pooling_size_real = pooling_size_real
        self.assign_ent = True
        self.max_num_nodes = max_num_nodes
        self.device = device
        self.edge_pool_weight = edge_pool_weight
        self.X_concat_Singular = X_concat_Singular
        self.pred_hidden_dims = pred_hidden_dims

        # # GC
        # self.conv_first_after_pool = nn.ModuleList()
        # self.conv_block_after_pool = nn.ModuleList()
        # self.conv_last_after_pool = nn.ModuleList()
        # for i in range(num_pooling):
        #     mul = 2 if self.X_concat_Singular else 1
        #     conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
        #         self.pred_input_dim * mul, hidden_dim, embedding_dim, num_layers,
        #         add_self, normalize=True, dropout=dropout)
        #     self.conv_first_after_pool.append(conv_first2)
        #     self.conv_block_after_pool.append(conv_block2)
        #     self.conv_last_after_pool.append(conv_last2)

        # EGAT
        self.DSN = DSN
        self.edgefeat_dim = 3
        self.use_simple_gat = use_simple_gat
        self.pred_input_dim = self.edgefeat_dim * embedding_dim
        self.egat_hidden_dims = egat_hidden_dims
        self.egat_dropout = egat_dropout
        self.egat_alpha = egat_alpha
        self.egat_num_heads = egat_num_heads

        print("Using EGAT edge feat")
        self.egat_list = nn.ModuleList()
        for i in range(num_pooling):
            mul = 2 if self.X_concat_Singular else 1
            if i == 0:
                # The first layer is still GNN, so we need to use the old pred_input_dim
                if use_simple_gat:
                    self.egat_list.append(
                        BatchGraphAttentionLayer(mul * (self.hidden_dim * (num_layers - 1) + embedding_dim) * 1,
                                                 embedding_dim, self.egat_dropout, self.egat_alpha).to(
                            device))
                else:
                    self.egat_list.append(
                        BatchGATFeatureExtractor(mul * (self.hidden_dim * (num_layers - 1) + embedding_dim) * 1,
                                                 self.edgefeat_dim, self.egat_hidden_dims, embedding_dim,
                                                 self.egat_dropout, self.egat_alpha, self.egat_num_heads).to(
                            device))
            else:
                if use_simple_gat:
                    self.egat_list.append(
                        BatchGraphAttentionLayer(self.pred_input_dim * 1, embedding_dim, self.egat_dropout,
                                                 self.egat_alpha).to(
                            device))
                else:
                    self.egat_list.append(
                        BatchGATFeatureExtractor(self.pred_input_dim * 1, self.edgefeat_dim, self.egat_hidden_dims,
                                                 embedding_dim, self.egat_dropout, self.egat_alpha,
                                                 self.egat_num_heads).to(
                            device))

        self.pred_model = self.build_pred_layers(
            self.hidden_dim * (self.num_layers - 1) + self.embedding_dim +
            self.pred_input_dim * self.num_pooling,
            self.pred_hidden_dims,
            self.label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, det_assign_matrix, det_edgefeat_list, det_edgepool_matrices_dic,
                **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            pool = Pool_Edges(det_edgepool_matrices_dic,
                              edge_pool_weight=self.edge_pool_weight,
                              X_concat_Singular=self.X_concat_Singular,
                              device=self.device)
            self.assign_tensor = det_assign_matrix
            self.assign_prob = det_assign_matrix
            assign_mask = self.construct_mask(self.assign_prob.shape[-1], batch_num_nodes_after_pool)

            coarsened_x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            x = pool(embedding_tensor, coarsened_x)

            # update pooled features and adj matrix
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            H = det_edgefeat_list
            # Concatenate H, H.T, A to get edge_feat
            edgefeat = torch.cat([adj.unsqueeze(-1), H.unsqueeze(-1), H.transpose(-1, -2).unsqueeze(-1)], dim=-1)
            if self.DSN:
                edgefeat = batch_DSN_fast(edgefeat)
            edgefeat_mask = (adj != 0).float()
            embedding_tensor = self.egat_forward(self.egat_list[i], x, edgefeat, assign_mask,
                                                 edgefeat_mask)

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def egat_forward(self, egat, embedding_tensor, edgefeat, embedding_mask, edgefeat_mask):
        edgefeat = torch.permute(edgefeat, (0, 3, 1, 2))
        if self.use_simple_gat:
            x_tensor, _ = egat(embedding_tensor, edgefeat, edgefeat_mask)
        else:
            x_tensor = egat(embedding_tensor, edgefeat, edgefeat_mask)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask

        return x_tensor

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(ICE_SVD_DET_SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(self.device))
            # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1 - adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss
