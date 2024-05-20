import pymetis
import numpy as np
from scipy.sparse import coo_matrix

class Graph:
    """A Graph with data associted at each vertex stored as 
        a weighted adjacency matrix using scipy.

    """
    def __init__(self, vertex_labels, adjacency_mat):
        self.labels = vertex_labels
        self.adjacency_mat = adjacency_mat

    def partition_metis(self, coarsening_factor):
        mat = self.adjacency_mat
        fine_n = mat.shape[0] 
        n_parts = int(fine_n / coarsening_factor)
        partition = pymetis.part_graph(n_parts, xadj=mat.indptr, adjncy=mat.indices, eweights=mat.data)

        p = coo_matrix((np.ones(fine_n), (np.arange(fine_n), partition[1])))
        coarse_mat = p.T @ mat @ p
        coarse_mat = coarse_mat.tocsr()

        return p, coarse_mat, partition

    def partition_metis_modularity(self, coarsening_factor):
        mat = self.adjacency_mat
        fine_n = mat.shape[0] 
        row_sums = mat @ np.ones(fine_n)
        t = row_sums.sum()

        mod_weights = np.ones(mat.nnz * 2, dtype=int)
        for edge_idx, i in enumerate(range(fine_n)):
            for j in mat.indices[mat.indptr[i]:mat.indptr[i+1]]:
                w = t - row_sums[i] * row_sums[j]
                if w > 0:
                    mod_weights[edge_idx] = w


        n_parts = int(fine_n / coarsening_factor)
        partition = pymetis.part_graph(n_parts, xadj=mat.indptr, adjncy=mat.indices, eweights=mod_weights)

        p = coo_matrix((np.ones(fine_n), (np.arange(fine_n), partition[1])))
        coarse_mat = p.T @ mat @ p
        coarse_mat = coarse_mat.tocsr()

        return p, coarse_mat, partition
