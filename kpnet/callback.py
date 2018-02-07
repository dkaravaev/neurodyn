import numpy as np
import networkx as nx
import scipy.stats

import kpnet.utils

class OutputCallback(object):
    def __init__(self, time_interval):
        self.result = np.zeros(shape=(1, time_interval))

    def compute(self, network, step):
        pass

class TotalActivityCallback(OutputCallback):
    def __init__(self, time_interval):
        super(TotalActivityCallback, self).__init__(time_interval)

    def compute(self, network, step):
        self.result[0, step] = np.sum(np.absolute(network.N))

class NeuronActivityCallback(OutputCallback):
    def __init__(self, time_interval, neuron_index):
        super(NeuronActivityCallback, self).__init__(time_interval)
        self.neuron_index = neuron_index

    def compute(self, network, step):
        self.result[0, step] = network.N[self.neuron_index]

class MeanNeuronActivityCallback(OutputCallback):
    def __init__(self, time_interval, mean_interval, neuron):
        super(MeanNeuronActivityCallback, self).__init__(int(time_interval / mean_interval))
        self.time_interval = time_interval
        self.mean_interval = mean_interval
        self.current = np.zeros(shape=(mean_interval, 1))
        self.neuron = neuron

    def compute(self, network, step):
        if step % self.mean_interval == 0:
            self.result[0, step / self.mean_interval] = np.mean(self.current)
        else:
            self.current[step % self.mean_interval] = network.N[self.neuron]

class TraceCallback(OutputCallback):
    def __init__(self, time_interval, norm, matrix_deg):
        super(TraceCallback, self).__init__(time_interval)
        self.norm = norm
        self.matrix_deg = matrix_deg

    def compute(self, network, step): 
        self.result[0, step] = np.trace(np.linalg.matrix_power(np.absolute(network.W0 / self.norm) > 0.5, self.matrix_deg))

class Weight0Callback(OutputCallback):
    def __init__(self, time_interval, neuron_1, neuron_2):
        super(Weight0Callback, self).__init__(time_interval)
        self.i = neuron_1
        self.j = neuron_2

    def compute(self, network, step):
        self.result[0, step] = network.W0[self.i, self.j]

class WeightCallback(OutputCallback):
    def __init__(self, time_interval, neuron_1, neuron_2):
        super(WeightCallback, self).__init__(time_interval)
        self.i = neuron_1
        self.j = neuron_2

    def compute(self, network, step):
        self.result[0, step] = network.W[self.i, self.j]

class PotentialCallback(OutputCallback):
    def __init__(self, time_interval, neuron):
        super(PotentialCallback, self).__init__(time_interval)
        self.i = neuron

    def compute(self, network, step):
        self.result[0, step] = network.P[self.i]

class EffectCallback(OutputCallback):
    def __init__(self, time_interval, neuron):
        super(EffectCallback, self).__init__(time_interval)
        self.i = neuron

    def compute(self, network, step):
        self.result[0, step] = network.x1[self.i] + network.x2[self.i]

class DegreeEntropyCallback(OutputCallback):
    def __init__(self, time_interval, threshold):
        super(DegreeEntropyCallback, self).__init__(time_interval)
        self.threshold = threshold

    def compute(self, network, step):
        norm = np.max(network.W0)
        if norm != 0:
            adj = np.absolute(network.W0 / norm) >= self.threshold
            probs = 1.0 * np.sum(adj, axis=1) / np.sum(adj)
            self.result[0, step] = scipy.stats.entropy(probs)
        else:
            self.result[0, step] = 0

class EingValues(OutputCallback):
    def __init__(self, time_interval, neuron):
        super(EingValues, self).__init__(time_interval)
        self.neuron = neuron

    def compute(self, network, step):
        self.result[0, step] = network._eig[self.neuron];

 
class AverageClusteringCoeff(OutputCallback):
    def __init__(self, time_interval, threshold):
        super(AverageClusteringCoeff, self).__init__(time_interval)
        self.threshold = threshold

    def compute(self, network, step):
        norm = np.max(network.W0)
        if norm != 0:
            adj = np.absolute(network.W0 / norm) >= self.threshold
            self.result[0, step] = np.mean(nx.clustering(nx.Graph(adj)).values())
        else:
            self.result[0, step] = 0
 
