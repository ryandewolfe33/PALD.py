import numpy as np
import sknetwork
import scipy.sparse
from itertools import product, combinations
import sknetwork.topology
from tqdm import tqdm


# Algorithm taken from https://www.pnas.org/doi/suppl/10.1073/pnas.2003634119#supplementary-materials suplementary information
def pald(D):
    n = D.shape[0]
    C = np.zeros_like(D, dtype="float32")
    with tqdm(total=n*(n-1)/2) as pbar:
        for i, j in tqdm(combinations(range(n), 2)):
            di = D[:, i]
            dj = D[:, j]
            uij = np.bitwise_or(di <= D[j,i], dj <= D[i,j])
            wi = (di[uij] < dj[uij]) + (di[uij] == dj[uij])/2
            C[i, uij] += wi/np.sum(uij)
            C[j, uij] += (1-wi)/np.sum(uij)
            pbar.update()
    C = C/(n-1)
    return C


"""
PALD algorithm for clustering vectors.
"""
class PALD:
    def fit(self, D):
        self.pald = pald(D)
        self.threshhold = np.mean(np.diagonal(self.pald))/2


    def predict(self):
        sym_C = np.minimum(self.pald, np.transpose(self.pald))
        strong_C = np.where(sym_C < self.threshhold, 0, sym_C)

        strong_G = scipy.sparse.csr_matrix(strong_C)
        self.clusters = sknetwork.topology.get_connected_components(strong_G)
        return self.clusters
    

    def fit_predict(self, D):
        self.fit(D)
        return self.predict()
