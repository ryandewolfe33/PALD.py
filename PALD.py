import numpy as np
import networkx as nx


class PALD:
    # Algorithm taken from https://www.pnas.org/doi/suppl/10.1073/pnas.2003634119#supplementary-materials suplementary information
    def _pald(self, D):
        n = D.shape[0]
        C = np.zeros_like(D)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                Uij = set()
                for k in range(n):
                    if D[k, i] <= D[i,j] or D[k, j] <= D[i,j]:
                        Uij.add(k)
                for l in Uij:
                    if D[l, i] < D[l,j]:
                        C[i, l] += 1/len(Uij)
                    if D[l, i] == D[l, j]:
                        C[i, l] += 1/(2*len(Uij))
        C = C/(n-1)
        self.pald = C
        return C
    

    def fit(self, D):
        self._pald(D)
        self.threshhold = np.mean(np.diagonal(self.pald))/2


    def predict(self):
        n = self.pald.shape[0]
        sym_C = np.minimum(self.pald, np.transpose(self.pald))
        strong_C = np.where(sym_C < self.threshhold, 0, sym_C)

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(1, n):
            for j in range(i):
                G.add_edge(i, j, weight=sym_C[i,j])
        self.G = G

        strong_G = nx.Graph()
        strong_G.add_nodes_from(range(n))
        for i in range(1, n):
            for j in range(i):
                if strong_C[i,j] > 0:
                    strong_G.add_edge(i, j, weight=strong_C[i,j])
        self.strong_G = strong_G

        components = nx.connected_components(strong_G)  # an iterator of sets
        clusters = np.full(len(strong_G.nodes), -1) # an array idx = id, value = cluster_id, -1 for outlier
        i = 0
        for comp in components:
            for n in comp:
                clusters[n] = i
            i += 1
        
        self.clusters = clusters
        return clusters
    

    def fit_predict(self, D):
        self.fit(D)
        return self.predict()