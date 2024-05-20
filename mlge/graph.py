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

        return p, coarse_mat
