import numpy as np
import networkx as nx
from itertools import product, combinations
from tqdm import tqdm


# Algorithm taken from https://www.pnas.org/doi/suppl/10.1073/pnas.2003634119#supplementary-materials suplementary information
def pald(D):
    n = D.shape[0]
    C = np.zeros_like(D)
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
Normalize a numpy array to sum to 1
Assumed non-negative
"""
def normalized(x):
    return x / np.sum(x)


"""
Get array of cluster ids, index is node_id, value is cluster_id
If G is not 0-n indexed must pass node_ids which is a dict node->index
"""
def cluster_from_graph(G, node_ids=None):
    components = nx.connected_components(G)  # an iterator of sets
    clusters = np.full(len(G.nodes), -1) # an array idx = id, value = cluster_id, -1 for outlier
    i = 0
    for component in components:
        for node_id in component:
            if node_ids:
                node_id = node_ids[node_id]
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
        self.G = G
        n = G.number_of_nodes()
        nodes = list(G.nodes)
        self.node_ids = {node: id for node, id in zip(nodes, range(n))}

        D = shortest_path_dissimilarity(G, self.node_ids)
        C = pald(D)

        neighbourhood_cohesion = np.zeros_like(D)
        for i, j in product(range(n), range(n)):
            wjt = np.delete(normalized(C[j, :]), i)
            Cit = np.delete(C[i, :], i)
            neighbourhood_cohesion[i,j] = np.dot(wjt, Cit)

            wit = np.delete(normalized(C[i, :]), j)
            Cjt = np.delete(C[j, :], j)
            neighbourhood_cohesion[j,i] = np.dot(wit, Cjt)
 
        dissipation = C - neighbourhood_cohesion
        Dii = np.tile(np.diagonal(dissipation), (n, 1)).transpose()
        Djj = np.tile(np.diagonal(dissipation), (n, 1))

        self.relative_dissipation = dissipation - (Dii + Djj)/2

        # Relative dissipation only exists for edges in G. Set other values to 0
        for i, j in product(range(n), range(n)):
            if (nodes[i], nodes[j]) not in G.edges and (nodes[j], nodes[i]) not in G.edges:    
                self.relative_dissipation[i,j] = self.relative_dissipation[j,i] = 0

        self.total_relative_dissipation = self.relative_dissipation + self.relative_dissipation.transpose()


    def predict(self):
        self.cluster_graph = nx.Graph()
        self.cluster_graph.add_nodes_from(self.G.nodes)

        nodes = {id:node for node, id in self.node_ids.items()}  # Reverse node_ids dict

        for i,j in zip(*np.nonzero(self.total_relative_dissipation)):
            # Only once per pair
            if i < j:
                continue
            if self.total_relative_dissipation[i,j] >= 0:
                continue
            node_i = nodes[i]
            node_j = nodes[j]
            self.cluster_graph.add_edge(node_i, node_j, weight=self.total_relative_dissipation[i,j])

        self.clusters = cluster_from_graph(self.cluster_graph, self.node_ids)
        self.predict = {node: self.clusters[id] for node, id in self.node_ids.items()}
        return self.predict


    def fit_predict(self, G):
        self.fit(G)
        return self.predict()
