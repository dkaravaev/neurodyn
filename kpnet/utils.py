import numpy as np

def sigmoid(x, gamma):
    return 1 / (1 + np.exp(-gamma * x))

def zero_diagonal(matrix):
    return matrix - np.diag(np.diag(matrix))

def moving_average(a, n=500) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n