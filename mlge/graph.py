import pymetis
import numpy as np
import scipy
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

class HyperGraph:
    """A Hypergaph with a weighted sparse matrix which relates
    hyperedges to vertices

    """

    def __init__(self, mat):
        self.mat = mat

    def disaggregate(self):
        """
        Created on Thu May 16 13:07:58 2024

        @author: panayot


         we disaggregate hypegraph into a standard graph by disaggregating  
         vertices that belong to several hyperedges  into several copies equal 
         to the number of hedges that contain them
         we create edges between these new disaggregated vertices if they 
         correspond to a same original vertex
         we also create new vertices one per hedge and connect that vertex with
         the disaggregated verteices coming from that hedge.
         in summary: new vertices equal  to the num-hedges + sum of all
         disaggregated vertices;
        """
        
        h_v = self.mat
        num_hedges = h_v.shape[0]
        v_h = h_v.T.tocsr()
        
        num_vertices = num_hedges + h_v.nnz
        i_vertex_vertex=np.zeros(num_vertices+1,int)
        
        vertex_to_v = np.zeros(num_vertices,int)
        
        for i in range(num_hedges):
            i_vertex_vertex[i] = h_v.indptr[i+1]-h_v.indptr[i]
        
        for i in range(num_hedges):
            for j in range(h_v.indptr[i],h_v.indptr[i+1]):
                v = h_v.indices[j]
                vertex = num_hedges + j
                vertex_to_v[vertex] = v
                i_vertex_vertex[vertex] = v_h.indptr[v+1] - v_h.indptr[v]
                
    # accumulate pointers:
        for i in range(num_vertices):
            i_vertex_vertex[i+1] += i_vertex_vertex[i]
        
        print('i_vertex_vertex[', num_vertices,']:', i_vertex_vertex[num_vertices])
        j_vertex_vertex = np.zeros(i_vertex_vertex[num_vertices], int)
        data_vertex_vertex = np.ones(i_vertex_vertex[num_vertices], float)

    # shift back pointers:
        for i in reversed(range(num_vertices)):
            i_vertex_vertex[i+1] = i_vertex_vertex[i]
            
        i_vertex_vertex[0] = 0
    #    print(num_vertices, i_vertex_vertex[num_vertices], h_v.indptr[num_hedges]+v_h.indptr[num_vs])
        
    # build disaggregated graph:
        for i in range(0,num_hedges):
            for j in range(h_v.indptr[i], h_v.indptr[i+1]):
                j_vertex_vertex[i_vertex_vertex[i]] = j + num_hedges
                i_vertex_vertex[i] += 1
                
        for i in range(0, num_hedges):
            for j in range(h_v.indptr[i], h_v.indptr[i+1]):
                v = h_v.indices[j]
                vertex = j + num_hedges
    #            print('vertex:',vertex, 'i_vertex_vertex[]:', i_vertex_vertex[vertex])
                j_vertex_vertex[i_vertex_vertex[vertex]] = i
                i_vertex_vertex[vertex] += 1
                for k in range(v_h.indptr[v], v_h.indptr[v+1]):
                    h = v_h.indices[k]
                    if (h != i):
                        for l in range(h_v.indptr[h], h_v.indptr[h+1]):
                            w = h_v.indices[l]
                            if (v == w):
                                j_vertex_vertex[i_vertex_vertex[vertex]] = l+num_hedges
                                i_vertex_vertex[vertex] += 1
                                
              

    # shift back pointers:
        for i in reversed(range(num_vertices)):
            i_vertex_vertex[i+1] = i_vertex_vertex[i]
        i_vertex_vertex[0] = 0
        
        
        
        vertex_vertex = scipy.sparse.csr_matrix((data_vertex_vertex, j_vertex_vertex, i_vertex_vertex), shape=(num_vertices, num_vertices))

        return vertex_vertex, vertex_to_v
