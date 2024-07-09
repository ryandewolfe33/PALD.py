import numpy as np
import networkx as nx
from itertools import product


# Algorithm taken from https://www.pnas.org/doi/suppl/10.1073/pnas.2003634119#supplementary-materials suplementary information
def pald(D):
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
    return C


"""
Normalize a numpy array to sum to 1
Assumed non-negative
taken from https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
"""
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


"""
Get array of cluster ids, index is node_id, value is cluster_id
G is assumed to have integer nodes with no gaps
"""
def cluster_from_graph(G):
    components = nx.connected_components(G)  # an iterator of sets
    clusters = np.full(len(G.nodes), -1) # an array idx = id, value = cluster_id, -1 for outlier
    i = 0
    for component in components:
        for node_id in component:
            clusters[node_id] = i
        i += 1
    return clusters


"""
PALD algorithm for clustering vectors.
"""
class PALD:
    def fit(self, D):
        self.pald = pald(D)
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
        self.clusters = cluster_from_graph(self.strong_G)
        return self.clusters
    

    def fit_predict(self, D):
        self.fit(D)
        return self.predict()
    

"""
Helper functions for GPALD
"""
def shortest_path_dissimilarity(G, node_ids):
    n = G.number_of_nodes()
    D = np.empty((n, n))
    path_lengths = nx.shortest_path_length(G)
    for source, path_dict in path_lengths:
        for target, distance in path_dict.items():
            source_id = node_ids[source]
            target_id = node_ids[target]
            D[source_id, target_id] = D[source_id, target_id] = distance
    return D


"""
PALD clustering for graph data.
"""
class GPALD:
    def __init__(self):
        pass


    def fit(self, G):
        n = G.number_of_nodes()
        nodes = list(G.nodes)
        self.node_ids = {node: id for node, id in zip(nodes, range(n))}

        D = shortest_path_dissimilarity(G, self.node_ids)
        C = pald(D)

        neighbourhood_cohesion = np.empty_like(D)
        for i, j in product(range(n), range(n)):
            wjt = normalized(np.delete(C[j, :], i))
            Cij = np.delete(C[i, :], i)
            neighbourhood_cohesion[i,j] = np.sum(wjt * Cij)
 
        dissipation = C - neighbourhood_cohesion

        Dii = np.tile(np.diagonal(dissipation), (n, 1))
        Djj = np.tile(np.diagonal(dissipation), (n, 1)).transpose()

        self.relative_dissipation = dissipation - (Dii + Djj)/2
        self.total_relative_dissipation = self.relative_dissipation + self.relative_dissipation.transpose()


    def predict(self):
        adjacency_matrix = np.where(self.total_relative_dissipation < 0, self.total_relative_dissipation, 0)
        cluster_graph = nx.from_numpy_array(np.abs(adjacency_matrix))
        clusters = cluster_from_graph(cluster_graph)

        out = dict()
        for node, id in self.node_ids.items():
            out[node] = clusters[id]
        return out


    def fit_predict(self, G):
        self.fit(G)
        return self.predict()
