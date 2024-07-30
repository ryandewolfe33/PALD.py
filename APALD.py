import scipy.sparse
import numpy as np
import pynndescent
import sknetwork.clustering
import leidenalg as la
import igraph as ig
from numba import jit


@jit
def size_of_union(a, b):
    return len(a) + len(b) - len(np.intersect1d(a, b))


@jit(nopython=True)  # Already optimial parralized
def make_other_knn_uxy_sizes(knn):
    other_sizes = np.empty_like(knn)
    cache = dict()
    for i in range(knn.shape[0]):
        for j in range(knn.shape[1]):
            if (i,j) in cache:
                other_sizes[i, j] = cache[(i,j)]
            else:
                size = size_of_union(knn[i, :], knn[knn[i,j], :])
                other_sizes[i, j] = size
                cache[(j, i)] = size
    return other_sizes


@jit(nopython=True, parallel=True)
def make_n_closer_than(knn, knn_dist):
    this_dist = np.empty((knn.shape[0], knn.shape[1], knn.shape[1]), dtype=knn_dist.dtype)
    for i in range(knn.shape[1]):
        this_dist[:, :, i] = knn_dist

    other_dist = np.empty_like(this_dist)
    for i in range(knn.shape[0]):
        other_dist[i, :, :] = knn_dist[knn[i, :], :]
    
    n_closer_than = np.sum(this_dist < other_dist, axis=2) + np.sum(this_dist == other_dist, axis=2)
    return n_closer_than


"""
The approximate local pald idea

Some graphs and some intuition suggest
a) local points have the highest cohesion
b) we only care about the cohesion of to local points for clustering (i.e. if a point is in a cluster it will be in said cluster with its nearest neighbours)

So, for some point x, y is random from it's knn, and then w is random from (knn_x or knn_y) as a set so double inclusion is not higher probability
"""
class APALD:
    """
    Create and APALD class and set some algorithm parameters.
    Parameters:
        n_neighbors=100: number of neighbors in the nearest neighbor graph.
        metric="cosine": the metric used for similarity. Anything supported by PyNNDescent is supported.
    """
    def __init__(self, n_neighbors=100, metric="cosine"):
        self.n_neighbors=n_neighbors
        self.metric=metric

    
    """
    Function to compute nearest neighbors and the APALD matrix.
    Parameters:
        data, a numpy array with rows for data points columns for data dimensions. 
    """
    def fit(self, data):
        self.index = pynndescent.NNDescent(data, metric=self.metric, n_neighbors=self.n_neighbors)
        knn = self.index.neighbor_graph[0]
        knn_dist = self.index.neighbor_graph[1]

        other_knn_uxy_sizes = make_other_knn_uxy_sizes(knn)
        n_closer_than = make_n_closer_than(knn, knn_dist)
        cohesion = n_closer_than / other_knn_uxy_sizes

        row = np.repeat(np.arange(knn.shape[0]), knn.shape[1])
        col = knn.flatten()
        data = cohesion.flatten()

        cohesion = scipy.sparse.coo_matrix((data, (row, col)))
        self.palds = cohesion.minimum(cohesion.transpose())

        
    """
    Function to predict the clusters from the APALD matrix generated in fit.
    Parameters:
        thresh=None, Can be a value 0 to 1 or "strong" or None. Cut edges from the APALD graph if below the threshhold. Strong refers to the original PALD paper threshhold of hald the mean diagonal.
        min_clustert_size=15: Any leiden clusters smaller than 15 are set as noise. Not required for algorithm to function.
    Returns:
        np.array of cluster assignments, -1 for noise.
    """
    def predict(self, thresh=0.5, min_cluster_size=15,):
        sym = self.palds.copy()
        if thresh=="strong":
            thresh = np.mean(sym.diagonal())/2
            sym.data = np.where(sym.data > thresh, sym.data, 0)
        elif thresh:
            sym.data = np.where(sym.data > np.quantile(sym.data, thresh), sym.data, 0)
        sym.setdiag(0)
        sym.eliminate_zeros()
        
        self.leiden = sknetwork.clustering.Leiden(n_aggregations=3)
        self.clusters = self.leiden.fit_predict(sym)

        return self.clusters
    

    """
    Convience Function
    """
    def fit_predict(self, data, thresh=None, min_cluster_size=15):
        self.fit(data)
        return self.predict(thresh=thresh, min_cluster_size=min_cluster_size)
