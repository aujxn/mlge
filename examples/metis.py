from scipy.io import mmread
from mlge.graph import Graph
import logging

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # cocktail-ingredient relation
    mat = mmread('../graph_data/cocktail.mtx').tocsr()

    with open('../graph_data/ingredient_labels.txt') as file:
        ingredient_labels = file.read().splitlines()

    #with open('cocktail_labels.txt') as file:
        #recipes = file.read().splitlines()

    # ingredient-ingredient relation
    adj_mat = mat.transpose() @ mat
    adj_mat = adj_mat.tocsr()
    n = adj_mat.shape[0]

    network = Graph(ingredient_labels, adj_mat)
    p, coarse_mat = network.partition_metis(2.0)
    print(f"Fine vertex count: {n}")
    print(f"Coarse vertex count: {p.shape[1]}")
    print(f"Fine nnz: {adj_mat.nnz}")
    print(f"Coarse nnz: {coarse_mat.nnz}")

if __name__ == "__main__":
    main()
