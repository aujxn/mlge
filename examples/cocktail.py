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
    # cocktail-ingredient relation
    mat = mmread('../graph_data/cocktail.mtx').tocoo()

    with open('../graph_data/ingredient_labels.txt') as file:
        ingredient_labels = file.read().splitlines()

    #with open('cocktail_labels.txt') as file:
        #recipes = file.read().splitlines()

    # ingredient-ingredient relation
    adj_mat = mat.transpose() @ mat
    n = adj_mat.shape[0]

    network = Graph(ingredient_labels, adj_mat)

    start_time = time.time()
    hierarchy = Hierarchy(network, 2.0)
    elapsed = time.time() - start_time
    logging.info(f"Built hierarchy with {len(hierarchy.adjacency_mats)} levels in {elapsed:.1f} seconds")

    colors = []
    coarsest_p = hierarchy.interpolation_mats[0]
    for p in hierarchy.interpolation_mats[1:]:
        coarsest_p = coarsest_p @ p

    for i in range(n):
        colors.append(coarsest_p.getrow(i).indices[0])

    start_time = time.time()
    embedding = Embedding(network, classes=colors, hierarchy=hierarchy)
    elapsed = time.time() - start_time
    embedding.embed_ml()
    logging.info(f"Built embedding in {elapsed:.1f} seconds")
    embedding.visualize()

if __name__ == "__main__":
    main()
