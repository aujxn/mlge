from scipy.io import mmread
from mlge.graph import Graph
from mlge.embed import Embedding
import logging

from mlge.hierarchy import Hierarchy

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    adj_mat = mmread('../graph_data/baseball.mtx').tocsr()

    with open('../graph_data/baseball_labels.txt') as file:
        teams = file.read().splitlines()

    n = adj_mat.shape[0]

    network = Graph(teams, adj_mat)
    p, coarse_mat, _ = network.partition_metis_modularity(2.0)
    print(f"Fine vertex count: {n}")
    print(f"Coarse vertex count: {p.shape[1]}")
    print(f"Fine nnz: {adj_mat.nnz}")
    print(f"Coarse nnz: {coarse_mat.nnz}")

    hierarchy = Hierarchy(network, 2.0, True)

    #embedding = Embedding(network, classes=partition[1], dim=3, gamma=1e2, beta=5e5, sigma=1e0, max_iter=100)
    embedding = Embedding(network, )
    coarsest_p = hierarchy.interpolation_mats[0]
    print(coarsest_p.shape)
    for p in hierarchy.interpolation_mats[1:]:
        print(p.shape)
        coarsest_p = coarsest_p @ p

    colors = []
    for i in range(n):
        colors.append(coarsest_p.getrow(i).indices[0])

    embedding = Embedding(network, classes=colors, hierarchy=hierarchy, max_iter=3000)
    embedding.embed()
    embedding.visualize()

if __name__ == "__main__":
    main()
