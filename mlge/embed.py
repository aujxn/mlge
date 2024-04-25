from scipy.sparse import csr_matrix
import numpy as np
from mlge.graph import Graph
# want to remove plotly eventually
import plotly.express as px
import logging

def unit_norm_random_vector(n):
   v = 2.0*np.random.random(n)-1.0
   v /= np.linalg.norm(v)
   return v


def embed(csr_mat: csr_matrix, dim: int, epsilon: float, m: int, max_iter: int):
    ''' Panayot's old embed code which isn't multilevel'''

    n = csr_mat.shape[0]
    indptr = csr_mat.indptr
    indices = csr_mat.indices

    x = np.random.rand(n, dim)
    for i in range(0,n):
        x[i] /= np.linalg.norm(x[i])
        x[i] *= np.random.random(1)

    x_old = np.zeros((n, dim))

    for iteration in range(max_iter):
        for i in range(n):
            direction = unit_norm_random_vector(dim)
            t_star = 0.0
            phi_min = 1.0/np.linalg.norm(x[i])+100.0/(1.0-np.linalg.norm(x[i]))
            for j in range(n):
                if (j != i):
                    phi_min+=10.0/np.linalg.norm(x[i]-x[j])
            for k in range(indptr[i], indptr[i+1]):
                j=indices[k]
                phi_min+=10000.0*np.linalg.norm(x[i]-x[j])
            a = -np.dot(x[i],direction)-np.sqrt(np.dot(x[i],direction)**2+1.0-np.linalg.norm(x[i])**2)
            b = -np.dot(x[i],direction)+np.sqrt(np.dot(x[i],direction)**2+1.0-np.linalg.norm(x[i])**2)
            # logging.debug('a:', a, 'b:',b, 'b-a:', b-a)
            if (b-a > epsilon):
                for k in range(1,m+1):
                    t= a + k*(b-a)/(m+1)
                    z= x[i]+t*direction
                    phi = 1.0/np.linalg.norm(z)+100.0/(1.0-np.linalg.norm(z))
                    for j in range(0,n):
                        if (j != i):
                            phi +=10.0/np.linalg.norm(z-x[j])
                    for l in range(indptr[i], indptr[i+1]):
                        j=indices[l]
                        phi += 10000.0*np.linalg.norm(z-x[j])
                    if (phi<phi_min):
                        t_star=t
                        phi_min = phi
                #logging.debug('t_star:', t_star)
                x[i]+=t_star*direction
        max_dist = 0.0
        for i in range(0,n):
            if (np.linalg.norm(x[i]-x_old[i]) > max_dist):
                max_dist = np.linalg.norm(x[i]-x_old[i])
        logging.debug(f"iter: {iteration}, max_dist: {max_dist:.2e}")
        for i in range(n):
            for k in range(dim):
                x_old[i][k] = x[i][k]

    return x
 
def visualize_plotly(positions, labels, truth=None):
    '''old visualization using plotly... probably want to move to pyvista'''

    n = len(positions)
    if truth is None:
        colors = [1]*n
    else:
        colors = truth 

    if positions.shape[1] == 2:
        fig = px.scatter(x=positions[:,0], y=positions[:,1], hover_name=labels, color=colors, size=[3]*n)
    elif positions.shape[1] == 3:
        fig = px.scatter_3d(x=positions[:,0], y=positions[:,1], z=positions[:,2], hover_name=labels, color=colors, size=[3]*n)
    else:
        print('Expected coordinates in 2d or 3d for visualization...')
        return

    fig.show()

class Embedding:

    def __init__(self, graph: Graph, dim=3, epsilon=1e-6, m=20, max_iter=50, classes=None):
        csr = graph.adjacency_mat
        self.graph = graph
        self.dim = dim 
        self.epsilon = epsilon 
        self.m = m 
        self.max_iter = max_iter
        self.positions = embed(csr, dim, epsilon, m, max_iter)
        if classes is not None:
            self.classes = classes

    def visualize(self):
        visualize_plotly(self.positions, self.graph.labels, truth=self.classes)
