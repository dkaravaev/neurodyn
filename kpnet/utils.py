import numpy as np

def sigmoid(x, gamma):
    return 1 / (1 + np.exp(-gamma * x))

def zero_diagonal(matrix):
    return matrix - np.diag(np.diag(matrix))

