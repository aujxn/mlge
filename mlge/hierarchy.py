from scipy.sparse import csc_matrix, eye
import numpy as np
from mlge.graph import Graph
import logging

def build_sparse_modularity_matrix(adj):
    ones = np.ones(adj.shape[0])
    row_sums = adj @ ones
    total = np.sum(row_sums)
    inverse_total = 1.0 / total
    mm = adj.copy()

    idx = 0
    for i, (row_start, row_end) in enumerate(zip(adj.indptr[:], adj.indptr[1:])):
        for j, weight in zip(adj.indices[row_start:row_end], adj.data[row_start:row_end]):
            mm.data[idx] = weight - inverse_total * row_sums[i] * row_sums[j]
            idx += 1

    mm.setdiag(0.0)
    mm.eliminate_zeros()
    return mm

'''
def sort_modularity(csr):
    result = []
    for row in range(csr.shape[0]):
        start_idx = csr.indptr[row]
        end_idx = csr.indptr[row + 1]
        data = csr.data[start_idx:end_idx]
        indices = csr.indices[start_idx:end_idx]

        filtered_sorted = sorted(
            [(d, i) for d, i in zip(data, indices) if d > 0 and i != row],
            key=lambda x: x[0],
            reverse=True
        )
        
        result.append(filtered_sorted)
    return result
'''

# This implementation is real inefficint and will not scale
def add_level(adj, cf):
    adj_coarse = adj
    n = adj.shape[0]
    p = eye(n)
    current_cf = 1.0

    while True:
        mm = build_sparse_modularity_matrix(adj_coarse)
        nc = adj_coarse.shape[0]
        wants_to_merge = np.full(nc, -1)

        for i in range(nc):
            row = mm.getrow(i)
            if len(row.data) > 0:
                target_idx = np.argmax(row.data)

                target = row.indices[target_idx]
                if target == i:
                    logging.error(f"vertex {i} wants to merge with self...")

                if row.data[target_idx] > 0:
                    wants_to_merge[i] = target


        pairs = []
        matches = np.zeros(nc)

        for i, j in enumerate(wants_to_merge):
            if j == -1:
                continue

            if i < j and wants_to_merge[j] == i:
                pairs.append((i,j))
                matches[i] = True
                matches[j] = True

        if len(pairs) == 0:
            return None

        agg_counter = 0
        row_idx = []
        col_idx = []

        for pair in pairs:
            row_idx.append(pair[0])
            row_idx.append(pair[1])
            col_idx.append(agg_counter)
            col_idx.append(agg_counter)
            agg_counter += 1

        #logging.debug(f"{len(pairs)} matches found")

        for i, matched in enumerate(matches):
            if not matched:
                row_idx.append(i)
                col_idx.append(agg_counter)
                agg_counter += 1

        data = np.ones(nc)
        new_p = csc_matrix((data, (row_idx, col_idx)), shape=(nc, agg_counter))
        adj_coarse = new_p.transpose() @ adj_coarse @ new_p
        p = p @ new_p
        current_cf = n / agg_counter

        if current_cf >= cf:
            return adj_coarse, p

        mm = build_sparse_modularity_matrix(adj_coarse)
        nc = agg_counter

def modularity_matching(adj, coarsening_factor):
    adjacency_mats = [adj]
    interpolation_mats = []
    logging.info(f"starting matching algorithm on network with {adj.shape[0]} vertices and {adj.nnz} edges")

    while True:
        result = add_level(adjacency_mats[-1], coarsening_factor)
        if result is not None:
            adj_coarse, p = result
            logging.info(f"Added level, coarse size: {adj_coarse.shape[0]} nnz: {adj_coarse.nnz}")
            adjacency_mats.append(adj_coarse)
            interpolation_mats.append(p)
        else:
            logging.info(f"Max modularity obtained with {len(adjacency_mats)} levels")
            return adjacency_mats, interpolation_mats

class Hierarchy:
    def __init__(self, graph: Graph, coarsening_factor, use_metis=False):

        if use_metis:
            self.adjacency_mats = [graph.adjacency_mat]
            self.interpolation_mats = []
            current_graph = graph
            for i in range(1):
                p, coarse_mat, _ = current_graph.partition_metis(2.0)
                # this might not work since diagonal needs to be removed for metis
                current_graph = Graph(None, coarse_mat)
                self.interpolation_mats.append(p)
                self.adjacency_mats.append(coarse_mat)
        else:
            coarse_mats, interpolations = modularity_matching(graph.adjacency_mat, coarsening_factor)
            self.adjacency_mats = coarse_mats
            self.interpolation_mats = interpolations
        


