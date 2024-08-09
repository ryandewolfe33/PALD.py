import scipy.sparse
import numpy as np
import pynndescent
import sknetwork.clustering
from numba import jit

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array

import warnings


@jit
def size_of_union(a, b):
    return len(a) + len(b) - len(np.intersect1d(a, b))


@jit(nopython=True)  # Already optimial parralized
def make_other_knn_uxy_sizes(knn):
    other_sizes = np.empty_like(knn)
    cache = dict()
    for i in range(knn.shape[0]):
        for j in range(knn.shape[1]):
            if (i, j) in cache:
                other_sizes[i, j] = cache[(i, j)]
            else:
                size = size_of_union(knn[i, :], knn[knn[i, j], :])
                other_sizes[i, j] = size
                cache[(j, i)] = size
    return other_sizes


@jit(nopython=True, parallel=True)
def make_n_closer_than(knn, knn_dist):
    this_dist = np.empty(
        (knn.shape[0], knn.shape[1], knn.shape[1]), dtype=knn_dist.dtype
    )
    for i in range(knn.shape[1]):
        this_dist[:, :, i] = knn_dist

    other_dist = np.empty_like(this_dist)
    for i in range(knn.shape[0]):
        other_dist[i, :, :] = knn_dist[knn[i, :], :]

    n_closer_than = np.sum(this_dist < other_dist, axis=2) + np.sum(
        this_dist == other_dist, axis=2
    )
    return n_closer_than


class PAKNNLD(ClusterMixin, BaseEstimator):
    """
    Partitioned K Nearest Neighbors Local Depth clustering algorithm. Based on the assertion that
    clustering is primarily a local problem, and uses on K nearest neighbors to compute cohesion
    values. Significantly faster than PALD, something like O(k^2 * n * log n) and suitable for hundreds
    thousands of points. Uses the Leiden clustering algorithm with modularity to cluster the cohesion
    graph (different than either paper).

    Extension of the paper "A social perspective
    on perceived distances reveals deep community structure" by Kenneth Berenhaut, Katherine
    Moorea, and Ryan Melvin,
    https://www.pnas.org/doi/10.1073/pnas.2003634119
    and mentioned in the paper "Partitioned K-nearest neighbor local depth for scalable
    comparison-based learning" by Baron, Darling, Davis, and Pettit.
    https://arxiv.org/abs/2108.08864

    Parameters
    ----------

    n_neighbors : int, default=100

    metric : string, default="cosine"
             Passed to pynndescent metric so a wide range of options are supported.

    thresh : float or string, default=0.5
             To cut down the cohesion graph for clustering, if thresh is a float remove
             the bottom thresh percentile of weights from the graph. If thresh="strong", drop
             edge weights below the PALD threshhold, equal to half the mean of the self cohesion.

    Attributes
    ----------

    labels_ : array-like of shape (n_samples,)
        An array of labels for the data samples; this is a integer array as per other scikit-learn
        clustering algorithms. A value of -1 indicates that a point is a noise point and
        not in any cluster.

    cohesion_ : array-like of shape (n_samples, n_samples)
        A matrix of the cohesion value C[x, w]. This matrix is not symmetric.
        Can be thought of a complete weighted directed graph on n_samples vertices.
    """

    def __init__(self, n_neighbors=100, metric="cosine", thresh=0.5):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.thresh = thresh

    def fit(self, X, y=None):
        """
        Fit the model to the data

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            The data to cluster.

        y : array-like of shape (n_samples,), default=None
            Ignored. This parameter exists only for compatibility with
            scikit-learn's fit_predict method.

        Returns
        -------

        self : PALD
               For compatibility with sklearn api.

        """
        X = check_array(X)
        n_neighbors = self.n_neighbors
        if n_neighbors > X.shape[0]:
            warnings.warn(
                "Asked for more neighbors that data points, defaulting to n_neighbors = n points."
            )
            n_neighbors = X.shape[0]

        index = pynndescent.NNDescent(X, metric=self.metric, n_neighbors=n_neighbors)
        knn = index.neighbor_graph[0]
        knn_dist = index.neighbor_graph[1]

        other_knn_uxy_sizes = make_other_knn_uxy_sizes(knn)
        n_closer_than = make_n_closer_than(knn, knn_dist)
        cohesion = n_closer_than / other_knn_uxy_sizes

        row = np.repeat(np.arange(knn.shape[0]), knn.shape[1])
        col = knn.flatten()
        data = cohesion.flatten()
        self.cohesion_ = scipy.sparse.coo_matrix((data, (row, col)))

        symmetric_cohesion = self.cohesion_.minimum(self.cohesion_.transpose())
        if self.thresh == "strong":
            thresh = np.mean(symmetric_cohesion.diagonal()) / 2
            symmetric_cohesion.data = np.where(
                symmetric_cohesion.data > thresh, symmetric_cohesion.data, 0
            )
        elif self.thresh:
            symmetric_cohesion.data = np.where(
                symmetric_cohesion.data
                > np.quantile(symmetric_cohesion.data, self.thresh),
                symmetric_cohesion.data,
                0,
            )
        symmetric_cohesion.setdiag(0)
        symmetric_cohesion.eliminate_zeros()

        # Check if matrix is empty
        if len(symmetric_cohesion.data) == 0:
            self.labels_ = np.arange(symmetric_cohesion.shape[0])
        else:
            leiden = sknetwork.clustering.Leiden(n_aggregations=3)
            self.labels_ = leiden.fit_predict(symmetric_cohesion)

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self
