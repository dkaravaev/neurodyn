import numpy as np

from kpnet.entropy import EntropyCWT

class ActivityCluster:
    def __init__(weights, thershold, min_size): 
        self.weights   = weights
        self.neurons   = weights.shape[0]
        self.thershold = thershold
        self.min_size  = min_size
   
    @property
    def clusters(self):
        clusters = []
        for i in range(self.neurons):
            encoded_clusters = self.neuron_cluster(i, thershold)
            clusters.append(self.parse_clusters(encoded_clusters, , i))
        

        return clusters

    def find_max_intersect(cluster, clusters, collected):
        max_len = 0
        n = -1
        c = -1
        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                if not collected[i][j]:
                    res = cluster & clusters[i][j]
                    if len(res) > max_len:
                        max_len = len(res)
                        n = i
                        c = j
        return n, c


    def neuron_cluster(neuron):
        cwts = [EntropyCWT(self.weights[neuron, j, :], Ricker()) for j in range(self.neurons)]
        entropies = np.zeros(shape=(self.neurons, self.neurons))
        for j in range(self.neurons):
            for k in range(self.neurons):
                entropies[j][k] = EntropyCWT.compare(cwts[j], cwts[k], 
                                  EntropyCWT.distribution, EntropyCWT.jensen)
        cluster = 1
        clusters = np.zeros(shape=(self.neurons, ))
        ents = np.inf + np.zeros(shape=(self.neurons, ))
        for j in range(neurons):
            if clusters[j] == 0:
                found = 0
                for k in range(neurons):
                    if k != j:
                        if entropies[j][k] <= np.min([ents[k], threshold]):
                            clusters[k] = cluster
                            ents[k] = entropies[j][k]
                        if entropies[j][k] <= np.min([ents[j], threshold]):
                            clusters[j] = cluster
                            ents[j] = entropies[j][k]
                c += cluster
                
        return cs

    def parse_clusters(encoded_clusters, neuron):
        clusters = []
        for num in np.unique(encoded_clusters):
            if num != 0:
                cluster = set()
                for j in range(N):
                    if encoded_clusters[j] == num:
                        cluster.add(j)
                if len(cluster) > self.min_size:
                    cluster.add(neuron)
                    clusters.append(cluster)
                
        return clusters
