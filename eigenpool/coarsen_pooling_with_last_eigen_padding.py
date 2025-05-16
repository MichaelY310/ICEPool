import graph 
import networkx 
import matplotlib.pyplot as plt
from networkx.algorithms import community
import community as cm
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy import sparse as sp
import scipy
import torch
from utils import sparse_mx_to_torch_sparse_tensor


def clusters_to_Entropy(A_sparse, clusters):
    num_clusters = len(clusters)
    H = scipy.sparse.lil_matrix((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            interconnection = A_sparse[clusters[i], :][:, clusters[j]]
            num_edges_from_i_to_j = interconnection.sum()
            p_ij = interconnection.sum(axis=1) / num_edges_from_i_to_j
            p_ji = interconnection.sum(axis=0) / num_edges_from_i_to_j
            p_ij = p_ij[np.where(p_ij > 0)]
            p_ji = p_ji[np.where(p_ji > 0)]
            H[i, j] = -(p_ij @ np.log(p_ij).T)
            H[j, i] = -(p_ji @ np.log(p_ji).T)
    return H


def DSN2(t):
	a = t.sum(dim=1, keepdim=True)
	b = t.sum(dim=0, keepdim=True)
	lamb = torch.cat([a.squeeze(), b.squeeze()], dim=0).max()
	r = t.shape[0] * lamb - t.sum(dim=0).sum(dim=0)

	a = a.expand(-1, t.shape[1])
	b = b.expand(t.shape[0], -1)
	tt = t + (lamb ** 2 - lamb * (a + b) + a * b) / r

	ttmatrix = tt / tt.sum(dim=0)[0]
	ttmatrix = torch.where(t > 0, ttmatrix, t)
	return ttmatrix


def DSN(x):
	"""Doubly stochastic normalization"""
	p = x.shape[0]
	y1 = []
	for i in range(p):
		y1.append(DSN2(x[i]))
	y1 = torch.stack(y1, dim=0)
	return y1


def adj2edgeindex(adj):
	adj = adj.tocoo().astype(np.float32)
	row = adj.row
	col = adj.col

	edge_index = torch.LongTensor([list(row),list(col)])

	return edge_index




class Graphs():
	def __init__(self, adjacency_matrix, pooling_sizes):
		self.adjacency_matrix = adjacency_matrix
		self.num_nodes = adjacency_matrix[:,0].shape[0]
		self.pooling_sizes = pooling_sizes
		self.graphs = [scipy.sparse.csr_matrix(adjacency_matrix)]
		self.layer2pooling_matrices=dict()
		self.edgefeats = []


	def coarsening_pooling(self, normalize = True, use_SVD=False):
		if use_SVD:
			self.layer2edge_pooling_matrices = dict()
			self.layer2edge_pooling_matrices_summed = dict()
		adj = scipy.sparse.csr_matrix(self.adjacency_matrix)
		for i in range(len(self.pooling_sizes)):
			if use_SVD:
				adj_coarsened, pooling_matrices, edge_pooling_matrices, edge_pooling_matrices_summed, edgefeat = self._coarserning_pooling_(adj, self.pooling_sizes[i], normalize, use_SVD)
			else:
				adj_coarsened, pooling_matrices, edgefeat = self._coarserning_pooling_(adj, self.pooling_sizes[i], normalize, use_SVD)
			self.graphs.append(adj_coarsened)
			self.layer2pooling_matrices[i] = pooling_matrices
			if use_SVD:
				self.layer2edge_pooling_matrices[i] = edge_pooling_matrices
				self.layer2edge_pooling_matrices_summed[i] = edge_pooling_matrices_summed
			self.edgefeats.append(edgefeat)
			adj = scipy.sparse.csr_matrix(adj_coarsened)


		num_nodes_before_final = adj_coarsened.shape[0]
		if num_nodes_before_final < 4:
			num_nodes_before_final = 4
		num_nodes_before_final = 4		
		pooling_matrices_final = [sp.lil_matrix((adj_coarsened.shape[0],1)) for i in range(num_nodes_before_final)]			
		if adj_coarsened.shape[0]>1:
			L_i = graph.laplacian(adj_coarsened, normalize)
			lamb_i, U_i = graph.fourier(L_i)

			for j in range(num_nodes_before_final):
				if j < adj_coarsened.shape[0]:
					if U_i[0,j] < 0:
						pooling_matrices_final[j][:,0] = -U_i[:,j].reshape(-1,1)
					else:
						pooling_matrices_final[j][:,0] = U_i[:,j].reshape(-1,1)
				else:
					if U_i[0, adj_coarsened.shape[0]-1] < 0:
						pooling_matrices_final[j][:,0] = -U_i[:, adj_coarsened.shape[0]-1].reshape(-1,1)
					else:
						pooling_matrices_final[j][:,0] = U_i[:, adj_coarsened.shape[0]-1].reshape(-1,1)

		else:
			for j in range(num_nodes_before_final):
				pooling_matrices_final[j][:,0] = adj_coarsened.reshape(-1,1)


		self.layer2pooling_matrices[i+1] = pooling_matrices_final

	def prepare_for_pytorch(self):
		self.edge_index_lists = [0]*len(self.graphs)
		for i in range(len(self.graphs)):
			self.edge_index_lists[i] = adj2edgeindex(self.graphs[i])
		for i in self.layer2pooling_matrices:
			self.layer2pooling_matrices[i] = [sparse_mx_to_torch_sparse_tensor(spmat).t() for spmat in self.layer2pooling_matrices[i]]







	def _coarserning_pooling_(self, adjacency_matrix, pooling_size, normalize=False, use_SVD=False):
		num_nodes = adjacency_matrix[:,0].shape[0]
		A_dense = adjacency_matrix.todense()
		num_clusters = int(num_nodes/pooling_size)
		if num_clusters == 0:
			num_clusters = num_clusters + 1
		sc = SpectralClustering(n_clusters = num_clusters, affinity= 'precomputed', n_init=10)
		sc.fit(np.array(A_dense))

		clusters = dict()
		for inx, label in enumerate(sc.labels_):
			if label not in clusters:
				clusters[label] = []
			clusters[label].append(inx)
		num_clusters = len(clusters)

		H = clusters_to_Entropy(adjacency_matrix, clusters).todense()

		num_nodes_in_largest_clusters = 0
		for label in clusters:
			if len(clusters[label])>=num_nodes_in_largest_clusters:

				num_nodes_in_largest_clusters = len(clusters[label])
		if num_nodes_in_largest_clusters <=5:
			num_nodes_in_largest_clusters = 5

		num_nodes_in_largest_clusters = 5 

		Adjacencies_per_cluster = [adjacency_matrix[clusters[label],:][:,clusters[label]] for label in range(len(clusters))]
######## Get inter matrix
		A_int = sp.lil_matrix(adjacency_matrix)

		for i in range(len(clusters)):
			zero_list = list(set(range(num_nodes)) - set(clusters[i]))
			for j in clusters[i]:
				A_int[j,zero_list] = 0
				A_int[zero_list,j] = 0


######## Getting adjacenccy matrix wuith only external links
		A_ext = adjacency_matrix - A_int
######## Getting cluster vertex indicate matrix

		row_inds = []
		col_inds = []
		data = []

		for i in clusters:
			for j in clusters[i]:
				row_inds.append(j)
				col_inds.append(i)
				data.append(1)

		Omega = sp.coo_matrix((data,(row_inds,col_inds)))
		A_coarsened = np.dot( np.dot(np.transpose(Omega),A_ext), Omega)

		np.expand_dims(A_coarsened.todense(), axis=-1)
		edgefeat = np.concatenate([np.expand_dims(A_coarsened.todense(), axis=-1), np.expand_dims(H, axis=-1), np.expand_dims(H.T, axis=-1)], axis=-1)

########## Constructing pooling matrix

		pooling_matrices = [sp.lil_matrix((num_nodes,num_clusters)) for i in range(num_nodes_in_largest_clusters)]

		for i in clusters:
			adj = Adjacencies_per_cluster[i]


			if len(clusters[i])>1:
				L_i = graph.laplacian(adj, normalize)
				lamb_i, U_i = graph.fourier(L_i)

				for j in range(num_nodes_in_largest_clusters):
					if j<len(clusters[i]): 
						if U_i[0,j] <0:
							pooling_matrices[j][clusters[i],i] = - U_i[:,j].reshape(-1,1)
						else:
							pooling_matrices[j][clusters[i],i] =  U_i[:,j].reshape(-1,1)
					else:
						if U_i[0, len(clusters[i])-1] <0:
							pooling_matrices[j][clusters[i],i] = - U_i[:, len(clusters[i])-1].reshape(-1,1)
						else:
							pooling_matrices[j][clusters[i],i] =  U_i[:, len(clusters[i])-1].reshape(-1,1)
			else:


				for j in range(num_nodes_in_largest_clusters):

					pooling_matrices[j][clusters[i],i] =  adj.reshape(-1,1)

		if not use_SVD:
			return A_coarsened, pooling_matrices, edgefeat







########## Edge Pooling
		edge_pooling_matrices = [[sp.lil_matrix((num_nodes, num_clusters)) for i in range(num_clusters)] for j in
								 range(3)]
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
			return A_coarsened, pooling_matrices, edge_pooling_matrices, edge_pooling_matrices_summed, edgefeat
		else:
			maximum_power = 3
			pow = min(maximum_power, round(np.sqrt(avg_cluster_size)))
			adjacency_matrix_powered = (
					scipy.sparse.linalg.matrix_power(adjacency_matrix, pow) + scipy.sparse.linalg.matrix_power(
				adjacency_matrix, pow - 1)).astype(bool).astype(float)
		edges = np.nonzero(adjacency_matrix_powered)
		edges = np.concatenate((edges[0][:, np.newaxis], edges[1][:, np.newaxis]), axis=-1)

		coarsened_connections = np.nonzero(
			A_coarsened)  # a tuple of two np.array, each with length num_edges*2, indicating source and target node_ids.
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
			num_singular_values_use = min(num_singular_values, 3)
			if num_singular_values_use < num_singular_values:
				U, Lambda, V = scipy.sparse.linalg.svds(A_ext_ij, k=num_singular_values_use)
				sorted_indices = np.argsort(-Lambda)
				Lambda = Lambda[sorted_indices]
				U = U[:, sorted_indices]
				V = V[sorted_indices, :]
			else:
				U, Lambda, V = np.linalg.svd(A_ext_ij.toarray())

			# print("connection_edges", connection_edges)
			# print("source_nodes_id", source_nodes_id)
			# print("target_nodes_id", target_nodes_id)
			# print("target_cluster_id", target_cluster_id)
			# print("source_cluster_id", source_cluster_id)
			# print("U", U.shape)
			# print("V", V.shape)
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






		return A_coarsened, pooling_matrices, edge_pooling_matrices, edge_pooling_matrices_summed, edgefeat




