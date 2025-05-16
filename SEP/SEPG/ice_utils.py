import torch
import numpy as np
from scipy import sparse as sp
import scipy

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


def toAdjMatrix(edges, size):
    A = torch.zeros(size, size)
    A[*edges] = 1
    return A.T


def toAssignMatrix(edges, size_from, size_to):
    # edges: 2 * N
    S = torch.zeros(size_to, size_from)
    S[*edges] = 1
    return S.T

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