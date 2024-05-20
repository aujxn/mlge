from scipy.sparse import csr_matrix
import numpy as np
from mlge.graph import Graph
from mlge.hierarchy import Hierarchy 
# want to remove plotly eventually
import plotly.express as px
import logging

def unit_norm_random_vector(n):
   v = 2.0*np.random.random(n)-1.0
   v /= np.linalg.norm(v)
   return v


def embed(csr_mat: csr_matrix, dim: int, epsilon: float, gamma: float, beta: float, sigma: float, m: int, max_iter: int):
    ''' Panayot's old embed code which isn't multilevel'''

    n = csr_mat.shape[0]
    indptr = csr_mat.indptr
    indices = csr_mat.indices
    data = csr_mat.data

    x = np.random.rand(n, dim)
    for i in range(0,n):
        x[i] /= np.linalg.norm(x[i])
        x[i] *= np.random.random(1)

    x_old = np.zeros((n, dim))

    for iteration in range(max_iter):
        for i in range(n):
            direction = unit_norm_random_vector(dim)
            t_star = 0.0
            phi_min = gamma * (1.0/np.linalg.norm(x[i])+10.0/(1.0-np.linalg.norm(x[i])))
            for j in range(n):
                if (j != i):
                    phi_min += sigma / np.linalg.norm(x[i]-x[j])**2.0
            for k in range(indptr[i], indptr[i+1]):
                j=indices[k]
                phi_min+= data[k] * beta * np.linalg.norm(x[i]-x[j])
            a = -np.dot(x[i],direction)-np.sqrt(np.dot(x[i],direction)**2+1.0-np.linalg.norm(x[i])**2)
            b = -np.dot(x[i],direction)+np.sqrt(np.dot(x[i],direction)**2+1.0-np.linalg.norm(x[i])**2)
            # logging.debug('a:', a, 'b:',b, 'b-a:', b-a)
            if (b-a > epsilon):
                for k in range(1,m+1):
                    t= a + k*(b-a)/(m+1)
                    z= x[i]+t*direction
                    phi = gamma * (1.0/np.linalg.norm(z)+10.0/(1.0-np.linalg.norm(z)))
                    for j in range(0,n):
                        if (j != i):
                            phi += sigma / np.linalg.norm(z-x[j])**2.0
                    for l in range(indptr[i], indptr[i+1]):
                        j=indices[l]
                        phi += data[l] * beta * np.linalg.norm(z-x[j])
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

'''
def multilevel_embed(csr_mat: csr_matrix, hierarchy: Hierarchy, dim: int, epsilon: float, gamma: float, beta: float, sigma: float, m: int, max_iter: int):
'''
''' New ML impl'''
'''
    n = csr_mat.shape[0]
    indptr = csr_mat.indptr
    indices = csr_mat.indices
    data = csr_mat.data

    x = np.random.rand(n, dim)
    for i in range(0,n):
        x[i] /= np.linalg.norm(x[i])
        x[i] *= np.random.random(1)

    x_old = np.zeros((n, dim))

    for iteration in range(max_iter):
        for i in range(n):
            direction = unit_norm_random_vector(dim)
            t_star = 0.0
            phi_min = gamma * (1.0/np.linalg.norm(x[i])+10.0/(1.0-np.linalg.norm(x[i])))
            for j in range(n):
                if (j != i):
                    phi_min += sigma / np.linalg.norm(x[i]-x[j])
            for k in range(indptr[i], indptr[i+1]):
                j=indices[k]
                phi_min += np.exp(data[k]) * beta * np.linalg.norm(x[i]-x[j])
            a = -np.dot(x[i],direction)-np.sqrt(np.dot(x[i],direction)**2+1.0-np.linalg.norm(x[i])**2)
            b = -np.dot(x[i],direction)+np.sqrt(np.dot(x[i],direction)**2+1.0-np.linalg.norm(x[i])**2)
            # logging.debug('a:', a, 'b:',b, 'b-a:', b-a)
            if (b-a > epsilon):
                for k in range(1,m+1):
                    t= a + k*(b-a)/(m+1)
                    z= x[i]+t*direction
                    phi = gamma * (1.0/np.linalg.norm(z)+10.0/(1.0-np.linalg.norm(z)))
                    for j in range(0,n):
                        if (j != i):
                            phi += sigma / np.linalg.norm(z-x[j])
                    for l in range(indptr[i], indptr[i+1]):
                        j=indices[l]
                        phi += np.exp(data[l]) * beta * np.linalg.norm(z-x[j])
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
    '''

def visualize_plotly(positions, labels, truth=None):
    '''old visualization using plotly... probably want to move to pyvista'''

    n = len(positions)
    if truth is None:
        colors = None
    else:
        colors = [str(val) for val in truth]

    if positions.shape[1] == 2:
        fig = px.scatter(x=positions[:,0], y=positions[:,1], hover_name=labels, color=colors, size=[3]*n)
    elif positions.shape[1] == 3:
        fig = px.scatter_3d(x=positions[:,0], y=positions[:,1], z=positions[:,2], hover_name=labels, color=colors, size=[3]*n)
    else:
        print('Expected coordinates in 2d or 3d for visualization...')
        return

    fig.show()

class Embedding:

    def __init__(self, graph: Graph, dim=3, eta=1e-5, gamma=1e1, beta=1e2, sigma=1e2, max_iter=10000, classes=None, hierarchy=None, delta=0.15):
        csr = graph.adjacency_mat
        self.graph = graph
        self.hierarchy = hierarchy
        self.dim = dim 
        self.eta = eta
        # don't need for new form
        #self.epsilon = epsilon 
        self.gamma = gamma 
        self.beta = beta 
        self.sigma = sigma
        self.delta = delta
        self.max_iter = max_iter
        n = graph.adjacency_mat.shape[0]
        x = np.random.rand(n, dim)
        for i in range(0,n):
            x[i] /= np.linalg.norm(x[i])
            x[i] *= np.random.random(1)
        self.positions = x
        #self.positions = embed(csr, dim, 1e-2, gamma, beta, sigma, 30, max_iter)
        if classes is not None:
            self.classes = classes

    def visualize(self):
        visualize_plotly(self.positions, self.graph.labels, truth=self.classes)

    def embed(self):
        for iteration in range(self.max_iter):
            update = self.get_update_directions(self.positions, 0)
            max_dist = 0.0
            for d in update:
                magnitude = np.linalg.norm(d) 
                max_dist = max(max_dist, magnitude)
            logging.debug(f"iter: {iteration}, max_dist: {max_dist:.2e}")
            self.positions += update

            if iteration % 1000 == 0:
                self.visualize()

    def embed_ml(self):
        for i in range(50):
            self.positions = self.v_cycle_recursive(self.positions, 3, 0)
            if i % 10 == 0:
                self.visualize()

    def v_cycle_recursive(self, x, steps, level):
        for iteration in range(steps * (level + 1)):
            update = self.get_update_directions(x, level)
            max_dist = 0.0
            for d in update:
                magnitude = np.linalg.norm(d) 
                if magnitude > 0.5:
                    d /= 2 * magnitude
                max_dist = max(max_dist, magnitude)
            logging.debug(f"level: {level}, forward iter: {iteration}, max_dist: {max_dist:.2e}")
            x += update

        if level < len(self.hierarchy.interpolation_mats):
            p = self.hierarchy.interpolation_mats[level]
            pt = p.T
            ones = np.ones(pt.shape[1])
            agg_sizes = pt @ ones
            xc = np.zeros((pt.shape[0], self.dim))
            for j in range(self.dim):
                xc[:,j] = (pt @ x[:,j]) / agg_sizes
            xc = self.v_cycle_recursive(xc, steps, level + 1)
            uc = p @ xc
            x = uc + self.delta * (x - uc)

            for iteration in range(steps * (level + 1)):
                update = self.get_update_directions(x, level)
                max_dist = 0.0
                for d in update:
                    magnitude = np.linalg.norm(d) 
                    if magnitude > 0.5:
                        d /= 2 * magnitude
                    max_dist = max(max_dist, magnitude)
                logging.debug(f"level: {level}, backward iter: {iteration}, max_dist: {max_dist:.2e}")
                x += update
        return x

    def get_update_directions(self, x, level):
        csr_mat = self.hierarchy.adjacency_mats[level]
        n = csr_mat.shape[0]
        indptr = csr_mat.indptr
        indices = csr_mat.indices
        data = csr_mat.data
        #x = self.positions
        dim = self.dim
    
        update = np.zeros((n, dim))
        
        for i in range(n):
            x_norm = np.linalg.norm(x[i])

            # ball and origin penalty
            if x_norm > 0.75:
                update[i] = self.gamma * (-1.0 * x_norm) * x[i]
            else:
                update[i] = self.gamma * 100.0 * (1.0 - x_norm**2.0) * x[i]

            #logging.debug(f'first update: {np.linalg.norm(update[i])}')
            # local geometric penalty (switch to octree for complexity scalability)
            for j in range(n):
                if (j != i):
                    dir = x[i] - x[j]
                    dist = np.linalg.norm(dir)
                    #if dist < 1e-10:
                        #dist = 1e-10
                    update[i] += (self.sigma / dist) * dir

            #logging.debug(f'update: {np.linalg.norm(update[i])}')
            # neighborhood penalty
            for idx in range(indptr[i], indptr[i+1]):
                j = indices[idx]
                dir = x[j] - x[i]
                update[i] += data[idx] * self.beta * np.linalg.norm(dir) * dir

            #logging.debug(f'update: {np.linalg.norm(update[i])}')

        return update * self.eta

