import scipy.sparse
import numpy as np
import pynndescent
import igraph as ig
import leidenalg as la

from itertools import product
from collections import Counter

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

        other_knn_uxy_sizes = np.empty_like(knn)
        for i, j in product(range(knn.shape[0]), range(knn.shape[1])):
            this = set(knn[i, :])
            other = set(knn[knn[i,j], :])
            other_knn_uxy_sizes[i, j] = len(this.union(other))

        this_dist = np.dstack([knn_dist]*knn.shape[1])

        other_dist = np.empty_like(this_dist)
        for i in range(knn.shape[0]):
            other_dist[i, :, :] = knn_dist[knn[i, :], :]

        n_closer_than = np.sum(this_dist < other_dist, axis=2) + np.sum(this_dist == other_dist, axis=2)
        cohesion = n_closer_than / other_knn_uxy_sizes

        row = np.repeat(range(knn.shape[0]), self.n_neighbors)
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
    def predict(self, thresh=0.5, min_cluster_size=15, leiden_method=la.ModularityVertexPartition):
        sym = self.palds.copy()
        if thresh=="strong":
            thresh = np.mean(sym.diagonal())/2
            sym.data = np.where(sym.data > thresh, sym.data, 0)
        elif thresh:
            sym.data = np.where(sym.data > np.quantile(sym.data, thresh), sym.data, 0)
        sym.setdiag(0)
        sym.eliminate_zeros()
        self.pald_graph = ig.Graph.Weighted_Adjacency(sym, mode="undirected")

        clusters = la.find_partition(self.pald_graph, leiden_method, weights="weight", n_iterations=5).membership

        if min_cluster_size > 1:
            cluster_sizes = Counter(clusters)
            for cluster, size in cluster_sizes.items():
                if size < min_cluster_size:
                    clusters[clusters == cluster] = -1
        
        self.clusters = clusters
        return self.clusters
    

    """
    Convience Function
    """
    def fit_predict(self, data, thresh=None, min_cluster_size=15, leiden_method=la.ModularityVertexPartition):
        self.fit(data)
        return self.predict(thresh=thresh, min_cluster_size=min_cluster_size, leiden_method=leiden_method)
