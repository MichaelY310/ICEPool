import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def DSN2(t):
    # print(t.shape)
    a = t.sum(dim=1, keepdim=True)
    b = t.sum(dim=0, keepdim=True)
    lamb = torch.cat([a.squeeze(1), b.squeeze(0)], dim=0).max()
    r = t.shape[0] * lamb - t.sum(dim=0).sum(dim=0)

    a = a.expand(-1, t.shape[1])
    b = b.expand(t.shape[0], -1)
    tt = t + (lamb ** 2 - lamb * (a + b) + a * b) / r

    ttmatrix = tt / tt.sum(dim=0)[0]
    ttmatrix = torch.where(t > 0, ttmatrix, t)
    return ttmatrix


"""Doubly stochastic normalization"""
# input: feat_dim * num_nodes * num_nodes
def DSN(x):
    p = x.shape[0]
    y1 = []
    for i in range(p):
        y1.append(DSN2(x[i]))
    y1 = torch.stack(y1, dim=0)
    return y1

# input: num_nodes * num_nodes * feat_dim
def DSN_fast(x):
    x_a = x.sum(dim=1, keepdim=True)
    x_b = x.sum(dim=0, keepdim=True)
    lamb, _ = torch.cat([x_a.squeeze(1), x_b.squeeze(0)], dim=0).max(dim=0)

    r = x.shape[-2] * lamb - x_a.squeeze(1).sum(-2)

    x_a = x_a.expand(x_a.shape[0], x.shape[0], x_a.shape[2])
    x_b = x_b.expand(x.shape[0], x_b.shape[1], x_b.shape[2])


    xx = x + (lamb**2 - lamb * (x_a+x_b) + x_a*x_b) / (r + 1e-20)

    xxmatrix = xx / (xx.sum(dim=0)[0, :] + 1e-20)
    xxmatrix = torch.where(x > 0, xxmatrix, x)

    return xxmatrix

# input: num_nodes * num_nodes * feat_dim
def batch_DSN_fast(x):
    bs = x.shape[0]
    feat_dim = x.shape[-1]
    x = x.permute(1, 2, 3, 0).reshape(x.shape[1], x.shape[2], -1)
    return DSN_fast(x).reshape(x.shape[0], x.shape[1], feat_dim, bs).permute(3, 0, 1, 2)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def forward(self, h, edge_attr, edge_attr_mask=None):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh) # num_nodes * num_nodes
        e = e * edge_attr  # edgefeat_dim * num_nodes * num_nodes
        zero_vec = -9e15 * torch.ones_like(e)
        if not edge_attr_mask:
            e = torch.where(edge_attr > 0, e, zero_vec)
        else:
            e = torch.where(edge_attr_mask == 1, e, zero_vec)
        e = F.softmax(e, dim=1)
        # e=torch.exp(e)

        # e=DSN(e)
        attention = F.dropout(e, self.dropout, training=self.training)

        h_prime = []
        for i in range(edge_attr.shape[0]):
            h_prime.append(torch.matmul(attention[i], Wh))

        if self.concat:
            h_prime = torch.cat(h_prime, dim=1)
            return F.elu(h_prime), e
        else:
            h_prime = torch.stack(h_prime, dim=0)
            h_prime = torch.sum(h_prime, dim=0)
            return h_prime

    # compute attention coefficient
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, ef_sz, nhid, nclass, dropout, alpha, nheads):
        """
        Dense version of GAT.
        nfeat输入节点的特征向量长度，标量
        ef_sz输入edge特征矩阵的大小，列表，PxNxN
        nhid隐藏节点的特征向量长度，标量
        nclass输出节点的特征向量长度，标量
        dropout：drpout的概率
        alpha：leakyrelu的第三象限斜率
        nheads：attention_head的个数
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 起始层
        self.attentions = [GraphAttentionLayer(nfeat, nhid[0], dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads[0])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # #hidden层
        # self.hidden_atts=[GraphAttentionLayer(nhid[0]*nheads[0]*ef_sz[0], nhid[1], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads[1])]
        # for i, attention in enumerate(self.hidden_atts):
        #     self.add_module('hidden_att_{}'.format(i), attention)

        # 输出层
        self.out_att = GraphAttentionLayer(nhid[0] * nheads[0] * ef_sz[0], nclass, dropout=dropout, alpha=alpha,
                                           concat=False)

    def forward(self, x, edge_attr):

        # 起始层
        x = F.dropout(x, self.dropout, training=self.training)  # 起始层
        temp_x = []
        for att in self.attentions:
            inn_x, edge_attr = att(x, edge_attr)
            temp_x.append(inn_x)
        x = torch.cat(temp_x, dim=1)  # 起始层

        # #中间层
        # x = F.dropout(x, self.dropout, training=self.training)#中间层
        # temp_x=[]
        # for att in self.hidden_atts:
        #     inn_x,edge_attr=att(x, edge_attr)
        #     temp_x.append(inn_x)
        # x = torch.cat(temp_x, dim=1)#中间层

        # 输出层
        x = F.dropout(x, self.dropout, training=self.training)  # 输出层
        x = F.elu(self.out_att(x, edge_attr))  # 输出层
        return F.log_softmax(x, dim=1)



class BatchGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(BatchGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    # edge_attr shape: (bs, edgefeat_dim, num_nodes, num_nodes)
    def forward(self, h, edge_attr, edge_attr_mask):
        edgefeat_dim = edge_attr.shape[1]
        Wh = h @ self.W
        e = self._prepare_attentional_mechanism_input(Wh) # num_nodes * num_nodes
        e = e * edge_attr.transpose(0, 1)
        e = e.transpose(0, 1)
        zero_vec = -9e15 * torch.ones_like(e)
        # e = torch.where(edge_attr > 0, e, zero_vec)
        edge_attr_mask = edge_attr_mask.unsqueeze(1).repeat(1, e.shape[1], 1, 1)
        e = torch.where(edge_attr_mask == 1, e, zero_vec)
        e = F.softmax(e, dim=1)

        attention = F.dropout(e, self.dropout, training=self.training)
        h_prime = torch.bmm(attention.reshape(attention.shape[0], -1, attention.shape[3]), Wh)
        h_prime = h_prime.reshape(h_prime.shape[0], edgefeat_dim, -1, h_prime.shape[2])
        h_prime = h_prime.transpose(0, 1)

        if self.concat:
            h_prime = torch.cat(torch.unbind(h_prime), dim=-1)
            return F.elu(h_prime), e
        else:
            h_prime = torch.sum(h_prime, dim=0)
            return h_prime

    # compute attention coefficient
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = Wh @ self.a[:self.out_features, :]
        Wh2 = Wh @ self.a[self.out_features:, :]
        # broadcast add
        e = Wh1 + Wh2.transpose(-2, -1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class BatchGAT(nn.Module):
    def __init__(self, nfeat, ef_sz, nhid, nclass, dropout, alpha, nheads):
        """
        Dense version of GAT.
        nfeat输入节点的特征向量长度，标量
        ef_sz输入edge特征矩阵的大小，列表，PxNxN
        nhid隐藏节点的特征向量长度，标量
        nclass输出节点的特征向量长度，标量
        dropout：drpout的概率
        alpha：leakyrelu的第三象限斜率
        nheads：attention_head的个数
        """
        super(BatchGAT, self).__init__()
        self.dropout = dropout

        # 起始层
        self.attentions = [BatchGraphAttentionLayer(nfeat, nhid[0], dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads[0])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # #hidden层
        # self.hidden_atts=[GraphAttentionLayer(nhid[0]*nheads[0]*ef_sz[0], nhid[1], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads[1])]
        # for i, attention in enumerate(self.hidden_atts):
        #     self.add_module('hidden_att_{}'.format(i), attention)

        # 输出层
        self.out_att = BatchGraphAttentionLayer(nhid[0] * nheads[0] * ef_sz[0], nclass, dropout=dropout, alpha=alpha,
                                           concat=False)

    def forward(self, x, edge_attr):

        # 起始层
        x = F.dropout(x, self.dropout, training=self.training)  # 起始层
        temp_x = []
        for att in self.attentions:
            inn_x, edge_attr = att(x, edge_attr)
            temp_x.append(inn_x)
        x = torch.cat(temp_x, dim=-1)  # 起始层

        # #中间层
        # x = F.dropout(x, self.dropout, training=self.training)#中间层
        # temp_x=[]
        # for att in self.hidden_atts:
        #     inn_x,edge_attr=att(x, edge_attr)
        #     temp_x.append(inn_x)
        # x = torch.cat(temp_x, dim=1)#中间层

        # 输出层
        x = F.dropout(x, self.dropout, training=self.training)  # 输出层
        x = F.elu(self.out_att(x, edge_attr))  # 输出层
        return F.log_softmax(x, dim=-1)


class BatchGATFeatureExtractor(nn.Module):
    # nheads[0] is the number of heads for the first layer
    # nheads[1] is the number of heads for the hidden layers
    def __init__(self, nfeat, edgefeat_dim, nhid, nclass, dropout, alpha, nheads, device="cuda"):
        """
        Dense version of GAT.
        nfeat输入节点的特征向量长度，标量
        ef_sz输入edge特征矩阵的大小，列表，PxNxN
        nhid隐藏节点的特征向量长度，标量
        nclass输出节点的特征向量长度，标量
        dropout：drpout的概率
        alpha：leakyrelu的第三象限斜率
        nheads：attention_head的个数
        """
        super(BatchGATFeatureExtractor, self).__init__()
        self.dropout = dropout

        print("=====================================")
        print("A BatchGATFeatureExtractor is Created")
        print(f"Input Attention Layer: input_dim: {nfeat}, out_dim: {edgefeat_dim}*{nheads[0]}*{nhid[0]}={edgefeat_dim*nheads[0]*nhid[0]}")
        for hid_i in range(len(nhid) - 1):
            if hid_i == 0:
                print(f"Hidden Attention Layer {hid_i}: input_dim: {edgefeat_dim}*{nheads[0]}*{nhid[hid_i]}={nhid[hid_i]*nheads[0]*edgefeat_dim}, out_dim: {edgefeat_dim}*{nheads[1]}*{nhid[hid_i+1]}={edgefeat_dim*nheads[1]*nhid[hid_i+1]}")
            else:
                print(f"Hidden Attention Layer {hid_i}: input_dim: {edgefeat_dim}*{nheads[1]}*{nhid[hid_i]}={nhid[hid_i]*nheads[1]*edgefeat_dim}, out_dim: {edgefeat_dim}*{nheads[1]}*{nhid[hid_i+1]}={edgefeat_dim*nheads[1]*nhid[hid_i+1]}")
        print(f"Final Attention Layer: input_dim: {nhid[-1]}*{nheads[1]}*{edgefeat_dim}={nhid[-1] * nheads[1] * edgefeat_dim}, out_dim: {edgefeat_dim}*{1}*{nclass}={edgefeat_dim*nclass}")
        print("=====================================")




        # 起始层
        self.attentions = [BatchGraphAttentionLayer(nfeat, nhid[0], dropout=dropout, alpha=alpha, concat=True).to(device) for _ in
                           range(nheads[0])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        #hidden层
        self.hidden_atts_list = []
        for hid_i in range(len(nhid) - 1):
            if hid_i == 0:
                hidden_atts = [BatchGraphAttentionLayer(nhid[hid_i]*nheads[0]*edgefeat_dim, nhid[hid_i+1], dropout=dropout, alpha=alpha, concat=True).to(device) for _ in range(nheads[1])]
                self.hidden_atts_list.append(hidden_atts)
            else:
                hidden_atts = [BatchGraphAttentionLayer(nhid[hid_i]*nheads[1]*edgefeat_dim, nhid[hid_i+1], dropout=dropout, alpha=alpha, concat=True).to(device) for _ in range(nheads[1])]
                self.hidden_atts_list.append(hidden_atts)

        # 输出层
        self.out_att = BatchGraphAttentionLayer(nhid[-1] * nheads[1] * edgefeat_dim, nclass, dropout=dropout, alpha=alpha,
                                           concat=True).to(device)

    def forward(self, x, edge_attr, edge_attr_mask):

        # 起始层
        x = F.dropout(x, self.dropout, training=self.training)  # 起始层
        temp_x = []
        for att in self.attentions:
            inn_x, edge_attr = att(x, edge_attr, edge_attr_mask)
            temp_x.append(inn_x)
        x = torch.cat(temp_x, dim=-1)  # 起始层

        #中间层
        for hidden_atts in self.hidden_atts_list:
            x = F.dropout(x, self.dropout, training=self.training)#中间层
            temp_x = []
            for att in hidden_atts:
                inn_x, edge_attr = att(x, edge_attr, edge_attr_mask)
                temp_x.append(inn_x)
            x = torch.cat(temp_x, dim=-1)#中间层

        # 输出层
        x = F.dropout(x, self.dropout, training=self.training)  # 输出层
        x = F.elu(self.out_att(x, edge_attr, edge_attr_mask)[0])  # 输出层
        return F.log_softmax(x, dim=-1)



# x = torch.tensor(np.random.randint(0, 10, (100, 8))).float()  # Shape: (30, 100, 8)
# e = torch.tensor(np.random.randint(0, 10, (3, 100, 100))).float()  # Shape: (30, 100, 100, 3)
#
# gat_model = GraphAttentionLayer(8, 16, 0.6, 0.2)
# a, b = gat_model(x, e)
# print(a.shape)
# print(b.shape)

# gat_model = GAT(nfeat, ef_sz, nhid, nclass, dropout, alpha, nheads)
# a, b = gat_model(x, e)
# print(a.shape)
# print(b.shape)



# x = torch.tensor(np.random.randint(0, 10, (30, 100, 8))).float()  # Shape: (30, 100, 8)
# e = torch.tensor(np.random.randint(0, 10, (30, 3, 100, 100))).float()  # Shape: (30, 100, 100, 3)
# mask = torch.tensor(np.random.randint(0, 2, (30, 100, 100))).float()

# gat_model = BatchGraphAttentionLayer(8, 16, 0.6, 0.2)
# a, b = gat_model(x, e)
# print(a.shape)
# print(b.shape)


# gat_model = BatchGATFeatureExtractor(8, 3, [32, 64, 64], 128, 0.6, 0.2, [1,3])
# a = gat_model(x, e, mask)
# print(a.shape)
