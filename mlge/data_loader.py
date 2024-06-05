import scipy
from mlge.graph import HyperGraph

class DataLoader:
    def __init__(self, root_dir="../graph_data"):
        self.root_dir = root_dir

    def contact_primary_school_classes(self):
        # TODO load labels / classes and such
        i = []
        j = []
        val = []

        with open(f"{self.root_dir}/contact-primary-school-classes/hyperedges-contact-primary-school-classes.txt") as file:
            for hyper_id, line in enumerate(file):
                for vertex_id in line.strip().split(','):
                    i.append(hyper_id)
                    j.append(int(vertex_id))
                    val.append(1.0)

        coo = scipy.sparse.coo_matrix((val, (i,j)))
        return HyperGraph(coo.tocsr())

