
from scipy.io import mmread
import numpy as np
import time
from mlge.graph import Graph
from mlge.hierarchy import Hierarchy
from mlge.embed import Embedding
import logging

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    mat = mmread('../graph_data/reddit.mtx').tocsr()
    adj_mat = mat + mat.T

    with open('../graph_data/reddit_labels.txt') as file:
        subreddits = file.read().splitlines()

    n = adj_mat.shape[0]
    network = Graph(subreddits, adj_mat)

    p, coarse_mat = network.partition_metis(2.0)
    print(f"Fine vertex count: {n}")
    print(f"Coarse vertex count: {p.shape[1]}")
    print(f"Fine nnz: {adj_mat.nnz}")
    print(f"Coarse nnz: {coarse_mat.nnz}")

if __name__ == "__main__":
    main()
