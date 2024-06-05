from scipy.io import mmwrite 
import numpy as np
from mlge.data_loader import DataLoader

def main():
    loader = DataLoader()
    matrix = "contact_primary_school"
    hypergraph = loader.contact_primary_school_classes()
    disagg, _ = hypergraph.disaggregate()
    mmwrite(f'../graph_data/{matrix}_disagg.mtx', disagg)

if __name__ == "__main__":
    main()
