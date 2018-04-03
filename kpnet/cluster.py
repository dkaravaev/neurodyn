import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class WTClustering:
    def __init__(self, W, pattern, distclass, wavelet):
        self.W = W
        self.N = self.W.shape[0]

        self.pattern   = pattern
        self.distclass = distclass
        self.wavelet   = wavelet

        self.pdist = self.distclass(pattern, self.wavelet)

        self.clusters = []
        

    @staticmethod
    def connected(dist1, dist2, epsilon, delta):
        ent_cond = dist1.jensen(dist2) <= epsilon 
        moda_cond = dist1.moda == dist2.moda
        dist_cond = np.abs(dist1.value[dist1.moda] - dist2.value[dist2.moda]) <= delta
        return ent_cond and moda_cond and dist_cond


    def to_adjmatrix(self):
        adj_matrix = np.zeros(shape=(self.N, self.N))
        for cluster in self.clusters:
            for elem1 in cluster:
                for elem2 in cluster:
                    if elem2 != elem1:
                        adj_matrix[elem2, elem1] = 1
        return adj_matrix 

    def run(self, begin, end, epsilon, delta):
        self.clusters = []
        visited = [False] * self.N
        for i in range(self.N):
            if not visited[i]:
                cluster = []
                for j in range(self.N):
                    if not visited[j] and j != i:
                        jdist = self.distclass(self.W[i, j, begin : end], self.wavelet)
                        if self.connected(jdist, self.pdist, epsilon, delta):
                            visited[j] = True
                            cluster.append(j)

                if len(cluster) > 0:
                    self.clusters.append(cluster + [i])
                    visited[i] = True
                    
        return self.clusters

    def plot(self, figsize):
        node_colors = [0.5] * self.N
        color_shift = 0.2
        for cluster in self.clusters:
            for elem in cluster:
                node_colors[elem] = color_shift
            color_shift += 0.2
        
        G = nx.from_numpy_matrix(self.to_adjmatrix())        
        plt.figure(figsize=figsize)
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color = node_colors, alpha = 0.7, node_size = 500)
        nx.draw_networkx_labels(G, pos, {i: str(i) for i in range(self.N)})
        nx.draw_networkx_edges(G, pos, edge_color='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()