import h5py

import numpy as np
import networkx as nx
import scipy.stats

import kpnet.utils


class OutputCallback(object):
    def __init__(self, time_interval):
        self.result = np.zeros(shape=(1, time_interval))
        self.time_interval = time_interval
        
    def compute(self, network, step):
        pass

class NormL1Callback(OutputCallback):
    def __init__(self, time_interval):
        super(NormL1Callback, self).__init__(time_interval)

    def compute(self, network, step):
        self.result[0, step] = np.sum(np.absolute(network.W0))

class NormL2Callback(OutputCallback):
    def __init__(self, time_interval):
        super(NormL2Callback, self).__init__(time_interval)

    def compute(self, network, step):
        self.result[0, step] = np.sum(np.absolute(network.W0 ** 2))

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
        self.result[0, step] = np.trace(np.linalg.matrix_power((network.W0 / self.norm) >= 0.8, self.matrix_deg))

class Weight0Callback(OutputCallback):
    def __init__(self, time_interval, neuron_1, neuron_2):
        super(Weight0Callback, self).__init__(time_interval)
        self.i = neuron_1
        self.j = neuron_2

    def compute(self, network, step):
        self.result[0, step] = network.W0[self.i, self.j]

class WeightsCallback(OutputCallback):
    def __init__(self, time_interval, neurons, chunk, filename):
        super(WeightsCallback, self).__init__(time_interval)
        self.chunk  = chunk
        self.slice  = np.zeros(shape=(neurons, neurons, chunk))
        self.f      = h5py.File(filename, 'w')
        self.dest   = self.f.create_dataset("W", (neurons, neurons, time_interval), dtype='float32')
        self.result = np.zeros(shape=(time_interval,))
        self.chunks = 0

    def compute(self, network, step):
        if (step % self.chunk == 0 and step != 0) or (step == self.time_interval - 1):
            self.dest[:, :, self.chunks * self.chunk : (self.chunks + 1) * self.chunk] = self.slice
            self.chunks += 1
        self.slice[:, :, step % self.chunk] = network.W0

    def __del__(self):
        self.f.close()
        
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

 
class ClusterCoeffCallback(OutputCallback):
    def __init__(self, time_interval):
        super(ClusterCoeffCallback, self).__init__(time_interval)

    def compute(self, network, step):
        M     = kpnet.utils.zero_diagonal(np.asarray((network.N.T * network.N) > 0.5, dtype='int32'))
        M2    = np.dot(M, M)
        M3    = np.dot(M, M2)
        num   = np.trace(M3)
        denom = 9 * (np.sum(M2) - np.trace(M2))
        if denom == 0:
            self.result[0, step] = 0
        else:
            self.result[0, step] = num / denom

class TotalDegreeCallback(OutputCallback):
    def __init__(self, time_interval):
        super(TotalDegreeCallback, self).__init__(time_interval)

    def compute(self, network, step):
        M = kpnet.utils.zero_diagonal(np.asarray((network.N.T * network.N) > 0.5, dtype='int32'))
        self.result[0, step] = np.trace(np.dot(M, M))

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