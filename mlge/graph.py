class Graph:
    """A Graph with data associted at each vertex stored as 
        a weighted adjacency matrix using scipy.

    """
    def __init__(self, vertex_labels, adjacency_mat):
        self.labels = vertex_labels
        self.adjacency_mat = adjacency_mat
