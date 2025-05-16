import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from scipy import sparse as sp
import scipy

import pickle
import random

from graph_sampler import GraphSampler
from torch_geometric.nn.pool import graclus
import os

# For pytorch data
def connection_entropy(A, S):
    A_ext = A - (S @ S.T) * A
    # SA[i, j] is the number of edges from node i to cluster j
    SA = S.T @ A_ext

    STA_extS = S.T @ A_ext @ S
    S_A = SA.unsqueeze(1)
    S_A = S_A * S.T.unsqueeze(0)
    P = S_A / (STA_extS.unsqueeze(-1) + 1e-20)
    H = -(P * torch.log(P + 1e-20)).sum(-1)

    return H.T

# For pytorch data
def batch_connection_entropy(A, S):
    A_ext = A - torch.bmm(S, S.transpose(-1, -2)) * A
    SA = torch.bmm(S.transpose(-1, -2), A_ext)

    STA_extS = torch.bmm(torch.bmm(S.transpose(-1, -2), A_ext), S)
    S_A = SA.unsqueeze(2)
    S_A = S_A * S.transpose(-1, -2).unsqueeze(1)
    P = S_A / (STA_extS.unsqueeze(-1) + 1e-20)
    H = -(P * torch.log(P + 1e-20)).sum(-1)

    return H.transpose(-1, -2)

def squeeze_clusters(clusters):
    d = 0
    correspondence = {}
    for i, cluster in enumerate(clusters):
        if not (cluster.item() in correspondence):
            correspondence[cluster.item()] = d
            d += 1
    return [correspondence[cluster.item()] for cluster in clusters], d

def toAssignMatrix(clusters, num_clusters):
    S = torch.zeros(len(clusters), num_clusters)
    S[torch.arange(len(clusters)), clusters] = 1
    return S

def apply_graclus(adj, edge_index, weights):
    clusters = graclus(edge_index, weight=weights, num_nodes=adj.shape[0])
    clusters, num_clusters = squeeze_clusters(clusters)
    assign_matrix = toAssignMatrix(clusters, num_clusters)
    adj = assign_matrix.T @ adj @ assign_matrix
    adj[torch.arange(adj.shape[0]), torch.arange(adj.shape[0])] = 0
    edge_index = torch.where(torch.triu(adj, diagonal=1))
    edge_index = torch.cat([edge_index[0].unsqueeze(0), edge_index[1].unsqueeze(0)], axis=0).int()
    weights = [adj[edge_index[0, i], edge_index[1, i]].int().item() for i in range(edge_index.shape[1])]

    return adj, edge_index, weights, assign_matrix

def iterative_graclus(G, assign_ratio, max_iters = 10):
    num_nodes = G.number_of_nodes()
    target_num_clusters = int(num_nodes * assign_ratio)

    adj = torch.tensor(nx.to_numpy_array(G)).float()
    edge_index = torch.tensor(list(G.edges())).T
    weights = torch.ones(edge_index.shape[1])

    assign_matrix = torch.eye(num_nodes)

    assign_matrices = []
    iters = 0
    while target_num_clusters < assign_matrix.shape[1] and iters < max_iters:
        assign_matrices.append(assign_matrix)
        iters += 1

        adj, edge_index, weights, assign_matrix = apply_graclus(
            adj,
            edge_index.long(),
            torch.tensor(weights).float()
        )

    assign_matrix = assign_matrices[0]
    for a in assign_matrices:
        assign_matrix @= a

    return assign_matrices, adj, assign_matrix

def shuffle_together(*args):
    zipped = list(zip(*args))
    random.shuffle(zipped)
    return [list(x) for x in zip(*zipped)]

def getEdgePoolMatrices(adj, assignment_matrix):
    num_nodes, num_clusters = assignment_matrix.shape

    adjacency_matrix = sp.csr_matrix(adj.numpy())
    assignment_matrix = assignment_matrix.numpy()

    clusters = []
    for cluster_id in range(assignment_matrix.shape[1]):
        clusters.append(np.nonzero(assignment_matrix[:, cluster_id])[0])

    A_coarsened = assignment_matrix.T @ adjacency_matrix @ assignment_matrix
    edge_pooling_matrices = [[sp.lil_matrix((num_nodes, num_clusters)) for i in range(num_clusters)] for j in
                             range(1)]
    # shape: num_components_stored * num_clusters * num_nodes * num_clusters
    # 1st dim: component
    # 2nd dim: from which cluster
    # 3rd dim: weight of the nodes in the from cluster that are connected to the to cluster
    # 4th dim: to which cluster
    # Did this because it seems like lil_matrix can't add in place. I will do summation operations afterwards.
    avg_cluster_size = int(num_nodes / num_clusters)
    if avg_cluster_size <= 1:
        edge_pooling_matrices_summed = []
        for x in edge_pooling_matrices:
            edge_pooling_matrices_summed.append(sum(x))
        return edge_pooling_matrices_summed
    else:
        maximum_power = 3
        pow = min(maximum_power, round(np.sqrt(avg_cluster_size)))
        adjacency_matrix_powered = (scipy.sparse.linalg.matrix_power(adjacency_matrix, pow) +
                                    scipy.sparse.linalg.matrix_power(adjacency_matrix, pow - 1)).astype(bool).astype(float)
    edges = np.nonzero(adjacency_matrix_powered)
    edges = np.concatenate((edges[0][:, np.newaxis], edges[1][:, np.newaxis]), axis=-1)

    coarsened_connections = np.nonzero(A_coarsened)  # a tuple of two np.array, each with length num_edges*2, indicating source and target node_ids.
    for i in range(coarsened_connections[0].shape[0]):
        source_cluster_id = coarsened_connections[0][i]
        target_cluster_id = coarsened_connections[1][i]
        if source_cluster_id > target_cluster_id:  # only consider the upper triangular part of the matrix
            continue
        id = np.where(
            np.isin(edges[:, 0], clusters[source_cluster_id]) * np.isin(edges[:, 1], clusters[target_cluster_id]))
        connection_edges = edges[id]
        source_nodes_id = np.unique(connection_edges[:, 0])
        target_nodes_id = np.unique(connection_edges[:, 1])
        A_ext_ij = adjacency_matrix_powered[source_nodes_id, :][:, target_nodes_id]
        num_singular_values = min(len(source_nodes_id), len(target_nodes_id))
        # num_singular_values_use = min(num_singular_values, 3)
        num_singular_values_use = 1
        if num_singular_values_use < num_singular_values:
            U, Lambda, V = scipy.sparse.linalg.svds(A_ext_ij, k=num_singular_values_use)
            sorted_indices = np.argsort(-Lambda)
            Lambda = Lambda[sorted_indices]
            U = U[:, sorted_indices]
            V = V[sorted_indices, :]
        else:
            U, Lambda, V = np.linalg.svd(A_ext_ij.toarray())


        V = V.T
        # print("num_singular_values_use", num_singular_values_use)
        for j in range(num_singular_values_use):
            # lil matrix raise error if you do something like lil_matrix[[0], 1] in which [0] has length only 1.
            if len(target_nodes_id) == 1:
                edge_pooling_matrices[j][target_cluster_id][target_nodes_id[0], source_cluster_id] = V[:,
                                                                                                     j] * np.sign(
                    V[0, j])
            else:
                # print(edge_pooling_matrices[j][target_cluster_id][target_nodes_id, source_cluster_id].shape)
                # print((V[:, j].reshape(-1, 1) * np.sign(V[0, j])).shape)
                edge_pooling_matrices[j][target_cluster_id][target_nodes_id, source_cluster_id] = V[:, j].reshape(
                    -1, 1) * np.sign(V[0, j])

            if len(source_nodes_id) == 1:
                edge_pooling_matrices[j][source_cluster_id][source_nodes_id[0], target_cluster_id] = U[:,
                                                                                                     j] * np.sign(
                    U[0, j])
            else:
                edge_pooling_matrices[j][source_cluster_id][source_nodes_id, target_cluster_id] = U[:, j].reshape(
                    -1, 1) * np.sign(U[0, j])

    edge_pooling_matrices_summed = []
    for x in edge_pooling_matrices:
        edge_pooling_matrices_summed.append(sum(x))

    return edge_pooling_matrices_summed

def prepare_val_data(graphs, args, val_idx, max_nodes=0):
    max_nodes = 0
    for g in graphs:
        max_nodes = max(max_nodes, g.number_of_nodes())




    # Preprocess data for DET methods
    if not os.path.exists(f"/scratch/yuanyang/ICE-results/DiffPool/preprocessed_data/{args.bmname}.pt"):
        # Precalculate Assignment Matrices
        print("Performing deterministic pooling...")
        all_assign_matrices = []
        all_adj_matrices = []
        # graphs = graphs[:10]
        for G in tqdm(graphs):
            _, _, assign_matrix = iterative_graclus(G, args.assign_ratio)
            all_assign_matrices.append(assign_matrix)
            all_adj_matrices.append(torch.tensor(nx.to_numpy_array(G)).float())

        max_num_clusters_fordet = 0
        for assign_matrix in all_assign_matrices:
            max_num_clusters_fordet = max(max_num_clusters_fordet, assign_matrix.shape[1])

        all_assign_matrices_padded = []
        all_adj_matrices_padded = []
        for assign_matrix, adj_matrix in zip(all_assign_matrices, all_adj_matrices):
            assign_matrix_padded = torch.zeros(max_nodes, max_num_clusters_fordet)
            assign_matrix_padded[:assign_matrix.shape[0], :assign_matrix.shape[1]] = assign_matrix
            all_assign_matrices_padded.append(assign_matrix_padded)

            adj_matrix_padded = torch.zeros(max_nodes, max_nodes)
            adj_matrix_padded[:adj_matrix.shape[0], :adj_matrix.shape[1]] = adj_matrix
            all_adj_matrices_padded.append(adj_matrix_padded)



        # Precalculate Connection Entropy
        print("Calculating Connection Entropy...")
        # A = torch.stack(all_adj_matrices_padded, dim=0)
        # S = torch.stack(all_assign_matrices_padded, dim=0)
        # all_edgefeat_lists = batch_connection_entropy(A, S)


        batch_size = 30
        all_results = []
        total_samples = len(all_adj_matrices_padded)
        num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division

        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            A_batch = torch.stack(all_adj_matrices_padded[start_idx:end_idx], dim=0)
            S_batch = torch.stack(all_assign_matrices_padded[start_idx:end_idx], dim=0)

            batch_result = batch_connection_entropy(A_batch, S_batch)
            all_results.append(batch_result)

        combined_result = torch.cat(all_results, dim=0)
        all_edgefeat_lists = combined_result




        # Precalculate edgepool matrices for SVD
        print("Calculating EdgePooling Matrices for SVD method...")
        all_edgepool_matrices_dics = []
        all_num_nodes_after_det_pool = []

        for i in tqdm(range(len(all_adj_matrices))):
            adj, assign_matrix = all_adj_matrices[i], all_assign_matrices[i]
            all_num_nodes_after_det_pool.append(assign_matrix.shape[1])
            edgepool_matrix = getEdgePoolMatrices(adj, assign_matrix)
            edgepool_matrix_padded = np.zeros((max_nodes, max_num_clusters_fordet))
            edgepool_matrix = edgepool_matrix[0].todense().astype(float)   # I only used one component here
            edgepool_matrix_padded[:edgepool_matrix.shape[0], :edgepool_matrix.shape[1]] = edgepool_matrix
            all_edgepool_matrices_dics.append(torch.tensor(edgepool_matrix_padded))

        all_assign_matrices = torch.stack(all_assign_matrices_padded, dim=0)
        all_edgepool_matrices_dics = torch.stack(all_edgepool_matrices_dics, dim=0)
        all_assign_matrices = all_assign_matrices.cpu()
        all_edgefeat_lists = all_edgefeat_lists.cpu()
        all_edgepool_matrices_dics = all_edgepool_matrices_dics.cpu()
        all_num_nodes_after_det_pool = all_num_nodes_after_det_pool.cpu()
        torch.save([all_assign_matrices, all_edgefeat_lists, all_edgepool_matrices_dics, all_num_nodes_after_det_pool], f"/scratch/yuanyang/ICE-results/DiffPool/preprocessed_data/{args.bmname}.pt")
        print("Deterministic pooling completed.")
        all_assign_matrices, all_edgefeat_lists, all_edgepool_matrices_dics, all_num_nodes_after_det_pool = torch.load(f"/scratch/yuanyang/ICE-results/DiffPool/preprocessed_data/{args.bmname}.pt")
    else:
        all_assign_matrices, all_edgefeat_lists, all_edgepool_matrices_dics, all_num_nodes_after_det_pool = torch.load(f"/scratch/yuanyang/ICE-results/DiffPool/preprocessed_data/{args.bmname}.pt")


    max_num_clusters_fordet = 0
    for assign_matrix in all_assign_matrices:
        max_num_clusters_fordet = max(max_num_clusters_fordet, assign_matrix.shape[1])

    graphs, all_assign_matrices, all_edgefeat_lists, all_edgepool_matrices_dics, all_num_nodes_after_det_pool = shuffle_together(graphs, all_assign_matrices, all_edgefeat_lists, all_edgepool_matrices_dics, all_num_nodes_after_det_pool)

    val_size = len(graphs) // 10
    train_graphs = graphs[:val_idx * val_size]
    train_assign_matrices = all_assign_matrices[:val_idx * val_size]
    train_edgefeat_lists = all_edgefeat_lists[:val_idx * val_size]
    train_edgepool_matrices_dics = all_edgepool_matrices_dics[:val_idx * val_size]
    train_num_nodes_after_det_pool = all_num_nodes_after_det_pool[:val_idx * val_size]

    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
        train_assign_matrices = train_assign_matrices + all_assign_matrices[(val_idx + 1) * val_size:]
        train_edgefeat_lists = train_edgefeat_lists + all_edgefeat_lists[(val_idx + 1) * val_size:]
        train_edgepool_matrices_dics = train_edgepool_matrices_dics + all_edgepool_matrices_dics[(val_idx + 1) * val_size:]
        train_num_nodes_after_det_pool = train_num_nodes_after_det_pool + all_num_nodes_after_det_pool[(val_idx + 1) * val_size:]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    val_assign_matrices = all_assign_matrices[val_idx*val_size: (val_idx+1)*val_size]
    val_edgefeat_lists = all_edgefeat_lists[val_idx*val_size: (val_idx+1)*val_size]
    val_edgepool_matrices_dics = all_edgepool_matrices_dics[val_idx*val_size: (val_idx+1)*val_size]
    val_num_nodes_after_det_pool = all_num_nodes_after_det_pool[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    max_nodes_use = max_nodes
    if args.method == 'soft-assign' or args.method == 'soft-assign-det': # To preserve the original diffpool implementation
        max_nodes_use = args.max_nodes
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes_use,
            features=args.feature_type, assign_matrices=train_assign_matrices,
            edgefeat_lists=train_edgefeat_lists, edgepool_matrices_dics=train_edgepool_matrices_dics, num_nodes_after_det_pool=train_num_nodes_after_det_pool)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes_use,
            features=args.feature_type, assign_matrices=val_assign_matrices,
            edgefeat_lists=val_edgefeat_lists, edgepool_matrices_dics=val_edgepool_matrices_dics, num_nodes_after_det_pool=val_num_nodes_after_det_pool)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, max_num_clusters_fordet, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim


def prepare_val_test_data(graphs, args, fold, max_nodes=0):
    max_nodes = 0
    for g in graphs:
        max_nodes = max(max_nodes, g.number_of_nodes())

    # Preprocess data for DET methods
    device = args.device
    if not os.path.exists(f"/scratch/yuanyang/ICE-results/DiffPool/preprocessed_data/{args.bmname}.pt"):
        # Precalculate Assignment Matrices
        print("Performing deterministic pooling...")
        all_assign_matrices = []
        all_adj_matrices = []

        for G in tqdm(graphs):
            _, _, assign_matrix = iterative_graclus(G, args.assign_ratio)
            all_assign_matrices.append(assign_matrix.to(device))
            all_adj_matrices.append(torch.tensor(nx.to_numpy_array(G)).float().to(device))

        max_num_clusters_fordet = max(assign_matrix.shape[1] for assign_matrix in all_assign_matrices)

        all_assign_matrices_padded = []
        all_adj_matrices_padded = []
        for assign_matrix, adj_matrix in zip(all_assign_matrices, all_adj_matrices):
            assign_matrix_padded = torch.zeros(max_nodes, max_num_clusters_fordet, device=device)
            assign_matrix_padded[:assign_matrix.shape[0], :assign_matrix.shape[1]] = assign_matrix
            all_assign_matrices_padded.append(assign_matrix_padded)

            adj_matrix_padded = torch.zeros(max_nodes, max_nodes, device=device)
            adj_matrix_padded[:adj_matrix.shape[0], :adj_matrix.shape[1]] = adj_matrix
            all_adj_matrices_padded.append(adj_matrix_padded)

        # Precalculate Connection Entropy
        print("Calculating Connection Entropy...")
        batch_size = 10
        all_results = []
        total_samples = len(all_adj_matrices_padded)
        num_batches = (total_samples + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            A_batch = torch.stack(all_adj_matrices_padded[start_idx:end_idx], dim=0).to(device)
            S_batch = torch.stack(all_assign_matrices_padded[start_idx:end_idx], dim=0).to(device)

            batch_result = batch_connection_entropy(A_batch, S_batch)
            all_results.append(batch_result.to(device))

        combined_result = torch.cat(all_results, dim=0)
        all_edgefeat_lists = combined_result

        # Precalculate edgepool matrices for SVD
        print("Calculating EdgePooling Matrices for SVD method...")
        all_edgepool_matrices_dics = []
        all_num_nodes_after_det_pool = []

        for i in tqdm(range(len(all_adj_matrices))):
            adj, assign_matrix = all_adj_matrices[i].cpu(), all_assign_matrices[i].cpu()
            all_num_nodes_after_det_pool.append(assign_matrix.shape[1])
            edgepool_matrix = getEdgePoolMatrices(adj, assign_matrix)
            edgepool_matrix_padded = np.zeros((max_nodes, max_num_clusters_fordet))
            edgepool_matrix = edgepool_matrix[0].todense().astype(float)
            edgepool_matrix_padded[:edgepool_matrix.shape[0], :edgepool_matrix.shape[1]] = edgepool_matrix
            all_edgepool_matrices_dics.append(torch.tensor(edgepool_matrix_padded, device=device))

        all_assign_matrices = torch.stack(all_assign_matrices_padded, dim=0).to(device)
        all_edgepool_matrices_dics = torch.stack(all_edgepool_matrices_dics, dim=0).to(device)

        torch.save([all_assign_matrices.cpu(), all_edgefeat_lists.cpu(), all_edgepool_matrices_dics.cpu(), all_num_nodes_after_det_pool],
                f"/scratch/yuanyang/ICE-results/DiffPool/preprocessed_data/{args.bmname}.pt")
        print("Deterministic pooling completed.")

    else:
        all_assign_matrices, all_edgefeat_lists, all_edgepool_matrices_dics, all_num_nodes_after_det_pool = torch.load(
            f"/scratch/yuanyang/ICE-results/DiffPool/preprocessed_data/{args.bmname}.pt")

    max_num_clusters_fordet = 0
    for assign_matrix in all_assign_matrices:
        max_num_clusters_fordet = max(max_num_clusters_fordet, assign_matrix.shape[1])

    graphs, all_assign_matrices, all_edgefeat_lists, all_edgepool_matrices_dics, all_num_nodes_after_det_pool = shuffle_together(
        graphs, all_assign_matrices, all_edgefeat_lists, all_edgepool_matrices_dics, all_num_nodes_after_det_pool)


    num_graphs = len(graphs)
    fold_size = num_graphs // 10
    indices = np.arange(num_graphs)
    test_start = fold * fold_size
    test_end = test_start + fold_size
    val_start = ((fold + 1) % 10) * fold_size
    val_end = val_start + fold_size
    test_idxes = indices[test_start:test_end]
    val_idxes = indices[val_start:val_end]
    train_idxes = np.setdiff1d(indices, np.concatenate((test_idxes, val_idxes)))

    train_graphs = [graphs[i] for i in train_idxes]
    train_assign_matrices = [all_assign_matrices[i] for i in train_idxes]
    train_edgefeat_lists = [all_edgefeat_lists[i] for i in train_idxes]
    train_edgepool_matrices_dics = [all_edgepool_matrices_dics[i] for i in train_idxes]
    train_num_nodes_after_det_pool = [all_num_nodes_after_det_pool[i] for i in train_idxes]

    val_graphs = [graphs[i] for i in val_idxes]
    val_assign_matrices = [all_assign_matrices[i] for i in val_idxes]
    val_edgefeat_lists = [all_edgefeat_lists[i] for i in val_idxes]
    val_edgepool_matrices_dics = [all_edgepool_matrices_dics[i] for i in val_idxes]
    val_num_nodes_after_det_pool = [all_num_nodes_after_det_pool[i] for i in val_idxes]

    test_graphs = [graphs[i] for i in test_idxes]
    test_assign_matrices = [all_assign_matrices[i] for i in test_idxes]
    test_edgefeat_lists = [all_edgefeat_lists[i] for i in test_idxes]
    test_edgepool_matrices_dics = [all_edgepool_matrices_dics[i] for i in test_idxes]
    test_num_nodes_after_det_pool = [all_num_nodes_after_det_pool[i] for i in test_idxes]








    

    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
          max([G.number_of_nodes() for G in graphs]), ', '
                                                      "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
          ', '
          "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    max_nodes_use = max_nodes
    if args.method == 'soft-assign' or args.method == 'soft-assign-det':  # To preserve the original diffpool implementation
        max_nodes_use = args.max_nodes
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes_use,
                                   features=args.feature_type, assign_matrices=train_assign_matrices,
                                   edgefeat_lists=train_edgefeat_lists,
                                   edgepool_matrices_dics=train_edgepool_matrices_dics,
                                   num_nodes_after_det_pool=train_num_nodes_after_det_pool)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes_use,
                                   features=args.feature_type, assign_matrices=val_assign_matrices,
                                   edgefeat_lists=val_edgefeat_lists, edgepool_matrices_dics=val_edgepool_matrices_dics,
                                   num_nodes_after_det_pool=val_num_nodes_after_det_pool)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes_use,
                                   features=args.feature_type, assign_matrices=test_assign_matrices,
                                   edgefeat_lists=test_edgefeat_lists,
                                   edgepool_matrices_dics=test_edgepool_matrices_dics,
                                   num_nodes_after_det_pool=test_num_nodes_after_det_pool)
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
        dataset_sampler.max_num_nodes, max_num_clusters_fordet, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim
