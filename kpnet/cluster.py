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
            clusters.append(self.parse_clusters(self.neuron_cluster(i, thershold), i, self.min_size))
        return clusters

    def neuron_cluster(self, neuron, thershold=0.01):
        c = 0
        cwts = [EntropyCWT(self.weights[neuron, j, :], Ricker()) for j in range(self.neurons)]
        cs = np.zeros(shape=(self.neurons, ))
        for j in range(self.neurons):
            for k in range(j):
                ent = EntropyCWT.compare(cwts[j], cwts[k], EntropyCWT.distribution, EntropyCWT.jensen)
                if ent <= thershold:
                    cs[k], cs[j] = c, c
            c += 1
        return cs

    def parse_clusters(self, encoded_clusters, neuron, min_size=1):
        clusters = []
        for num in np.unique(encoded_clusters):
            cluster = set()
            for j in range(self.neurons):
                if encoded_clusters[j] == num:
                    cluster.add(j)
            if len(cluster) > min_size:
                cluster.add(neuron)
                clusters.append(cluster)
                
        return clusters
