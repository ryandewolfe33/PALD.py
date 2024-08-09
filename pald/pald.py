import numpy as np
import sknetwork
import scipy.sparse
from sklearn.metrics import pairwise_distances

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array

from itertools import combinations

"""
Compute the C[x, w] cohesion matrix. Algorithm taken from
https://www.pnas.org/doi/suppl/10.1073/pnas.2003634119#supplementary-materials
"""
def cohesion(D):
    n = D.shape[0]
    C = np.zeros_like(D, dtype="float32")
    for i, j in combinations(range(n), 2):
        di = D[:, i]
        dj = D[:, j]
        uij = np.bitwise_or(di <= D[j,i], dj <= D[i,j])
        wi = (di[uij] < dj[uij]) + (di[uij] == dj[uij])/2
        C[i, uij] += wi/np.sum(uij)
        C[j, uij] += (1-wi)/np.sum(uij)
    C = C/(n-1)
    return C


class PALD(ClusterMixin, BaseEstimator):
    """
    Partitioned Local Depth clustering algorithm. Based on comparisons between distances,
    so scales decently to high dimensions. O(n^3) run time and O(n^2) space requirements so
    not reccomended for more than a few thousand points. Based on the paper "A social perspective
    on perceived distances reveals deep community structure" by Kenneth Berenhaut, Katherine
    Moorea, and Ryan Melvin. 
    https://www.pnas.org/doi/10.1073/pnas.2003634119

    Parameters
    ----------

    metric : string, default="euclidean"
             Passed to sklearn.metrics.pariwise_distances so a wide range of options are supported.

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
    
    def __init__(self, metric="euclidean"):
        self.metric = metric


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
        if self.metric == "precomputed":
            X = check_array(X, force_all_finite=False)
            D = X
        else:
            X = check_array(X)
            D = pairwise_distances(X, metric=self.metric)

        self.cohesion_ = cohesion(D)
        threshhold = np.mean(np.diagonal(self.cohesion_))/2
        symmetric_cohesion = np.minimum(self.cohesion_, self.cohesion_.T)
        strong_cohesion = np.where(symmetric_cohesion < threshhold, 0, symmetric_cohesion)
        strong_graph = scipy.sparse.csr_matrix(strong_cohesion)
        self.labels_ = sknetwork.topology.get_connected_components(strong_graph)

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        
        return self
