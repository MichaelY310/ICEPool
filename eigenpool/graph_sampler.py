import networkx as nx
import numpy as np
import torch
import torch.utils.data
from sklearn import preprocessing


def DSN2(t):
    # print(t.shape)
    a = t.sum(dim=1, keepdim=True)
    b = t.sum(dim=0, keepdim=True)
    # print("===========")
    # print(a.shape)
    # print(b.shape)
    # print(a.squeeze())
    # print(b.squeeze())
    # # print(torch.cat([a.squeeze(), b.squeeze()]))
    # print("===========")
    lamb = torch.cat([a.squeeze(1), b.squeeze(0)], dim=0).max()
    # lamb = torch.tensor(max(a.max(), b.max()), dtype=a.dtype).to(a.device)
    # print(max(a.max(), b.max()))
    # print(lamb)
    r = t.shape[0] * lamb - t.sum(dim=0).sum(dim=0)

    a = a.expand(-1, t.shape[1])
    b = b.expand(t.shape[0], -1)
    tt = t + (lamb ** 2 - lamb * (a + b) + a * b) / r

    ttmatrix = tt / tt.sum(dim=0)[0]
    ttmatrix = torch.where(t > 0, ttmatrix, t)
    return ttmatrix


"""Doubly stochastic normalization"""


def DSN(x):
    # print(x)
    # print(x.shape)
    p = x.shape[0]
    y1 = []
    for i in range(p):
        y1.append(DSN2(x[i]))
    y1 = torch.stack(y1, dim=0)
    return y1


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


class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''

    def __init__(self, G_list, graphs_list, num_pool_matrix, num_pool_final_matrix, features='default', normalize=True,
                 assign_feat='default', max_num_nodes=0, norm='l2', DSN=True):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        self.graphs_list = graphs_list
        self.num_pool_matrix = num_pool_matrix
        self.num_pool_final_matrix = num_pool_final_matrix
        self.norm = norm
        self.DSN = DSN

        self.assign_feat_all = []

        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        self.feat_dim = G_list[0]._node[0]['feat'].shape[0]

        for G in G_list:
            adj = np.array(nx.to_numpy_array(G))
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i, u in enumerate(G.nodes()):
                    f[i, :] = G._node[u]['feat']
                self.feature_all.append(f)
            elif features == 'id':
                self.feature_all.append(np.identity(self.max_num_nodes))
            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'deg':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs > max_deg] = max_deg
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                feat = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                              'constant', constant_values=0)

                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i, u in enumerate(G.nodes()):
                    f[i, :] = G._node[u]['feat']

                feat = np.concatenate((feat, f), axis=1)

                self.feature_all.append(feat)
            elif features == 'struct':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs > 10] = 10
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                degs = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                              'constant', constant_values=0)

                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings,
                                                    [0, self.max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                g_feat = np.hstack([degs, clusterings])
                if 'feat' in G._node[0]:
                    node_feats = np.array([G._node[i]['feat'] for i in range(G.number_of_nodes())])
                    node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                                        'constant')
                    g_feat = np.hstack([g_feat, node_feats])

                self.feature_all.append(g_feat)

            # print('feature shapoe 1..1.', self.feature_all[0].shape)

            if assign_feat == 'id':
                self.assign_feat_all.append(
                    np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])))
            else:
                self.assign_feat_all.append(self.feature_all[-1])

        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        graph = self.graphs_list[idx]

        return_dic = {'adj': adj_padded,
                      'feats': self.feature_all[idx].copy(),
                      'label': self.label_all[idx],
                      'num_nodes': num_nodes,
                      'assign_feats': self.assign_feat_all[idx].copy()}

        for i in range(len(graph.graphs) - 1):
            ind = i + 1
            adj_key = 'adj_pool_' + str(ind)
            num_nodes_key = 'num_nodes_' + str(ind)
            num_nodes_ = graph.graphs[ind].shape[0]
            return_dic[num_nodes_key] = num_nodes_
            adj_padded_ = np.zeros((self.max_num_nodes, self.max_num_nodes))
            adj_padded_[:num_nodes_, :num_nodes_] = graph.graphs[ind].todense().astype(float)
            return_dic[adj_key] = adj_padded_

            edgefeat_padded_ = np.zeros((self.max_num_nodes, self.max_num_nodes, graph.edgefeats[i].shape[-1]))

            ########## DSN in EGAT
            if self.DSN:
                # print(torch.tensor(graph.edgefeats[i].astype(float)).permute(2, 0, 1))
                processed_edgefeat = DSN(torch.tensor(graph.edgefeats[i].astype(float)).permute(2, 0, 1)).permute(1, 2, 0)
                processed_edgefeat = torch.where(torch.isnan(processed_edgefeat), torch.tensor(1.0), processed_edgefeat)
                # if (torch.any(torch.isnan(processed_edgefeat))):
                #     print(graph.edgefeats[i])
                # processed_edgefeat = DSN(torch.tensor(graph.edgefeats[i].astype(float)))
            else:
                processed_edgefeat = torch.tensor(graph.edgefeats[i].astype(float))

            edgefeat_padded_[:num_nodes_, :num_nodes_, :] = processed_edgefeat
            return_dic['edgefeat_' + str(i)] = edgefeat_padded_
            edgefeat_mask_padded_ = np.zeros((self.max_num_nodes, self.max_num_nodes))
            edgefeat_mask_padded_[:num_nodes_, :num_nodes_] = graph.graphs[ind].todense().astype(bool).astype(float)
            return_dic['edgefeat_mask_' + str(i)] = edgefeat_mask_padded_

        for i in range(len(graph.layer2pooling_matrices)):
            if i == len(graph.layer2pooling_matrices) - 1:
                for j in range(self.num_pool_final_matrix):
                    pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)
                    pool_adj = graph.layer2pooling_matrices[i][j]
                    pool_adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
                    if self.norm == 'l1':
                        pool_adj = pool_adj.todense().astype(float)
                        pool_adj = preprocessing.normalize(pool_adj, norm=self.norm, axis=0)
                        pool_adj_padded[:pool_adj.shape[0], : pool_adj.shape[1]] = pool_adj
                    else:
                        pool_adj_padded[:pool_adj.shape[0], : pool_adj.shape[1]] = pool_adj.todense().astype(float)
                    return_dic[pool_adj_key] = pool_adj_padded
            else:
                for j in range(self.num_pool_matrix):
                    pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)
                    pool_adj = graph.layer2pooling_matrices[i][j]
                    pool_adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
                    if self.norm == 'l1':
                        pool_adj = pool_adj.todense().astype(float)
                        pool_adj = preprocessing.normalize(pool_adj, norm=self.norm, axis=0)
                        pool_adj_padded[:pool_adj.shape[0], : pool_adj.shape[1]] = pool_adj
                    else:
                        pool_adj_padded[:pool_adj.shape[0], : pool_adj.shape[1]] = pool_adj.todense().astype(float)
                    return_dic[pool_adj_key] = pool_adj_padded

        if hasattr(graph, "layer2edge_pooling_matrices_summed"):
            for i in range(len(graph.layer2edge_pooling_matrices_summed)):
                num_pool_matrix = self.num_pool_matrix if i != len(
                    graph.layer2pooling_matrices) - 1 else self.num_pool_final_matrix
                for j in range(num_pool_matrix):
                    edge_pool_adj_summed_key = 'edge_pool_adj_summed_' + str(i) + '_' + str(j)
                    edge_pool_adj_summed = graph.layer2edge_pooling_matrices_summed[i][j]
                    edge_pool_adj_padded_summed = np.zeros((self.max_num_nodes, self.max_num_nodes))
                    edge_pool_adj_padded_summed[:edge_pool_adj_summed.shape[0], :edge_pool_adj_summed.shape[1]] = edge_pool_adj_summed.todense().astype(float)
                    return_dic[edge_pool_adj_summed_key] = edge_pool_adj_padded_summed


        return return_dic
