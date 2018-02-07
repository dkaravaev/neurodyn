import kpnet.utils
import numpy as np

from scipy.stats import norm


class KPNetwork(object):
    A1, B1, C1 = 0.4, 0.2, 0.2    
    A2, B2, C2 = 0.2, 0.5, 0.1

    mu, nu = 0.001, 0.1
    
    def __init__(self, n, alpha=0.2, beta=1.0):
        self.n = n
        self.alpha, self.beta = alpha, beta
        
        self.W, self.W0, self.h  = np.zeros(shape=(n, n)), np.zeros(shape=(n, n)), np.zeros(shape=(n, 1))
        self.P, self.x1, self.x2 = np.zeros(shape=(n, 1)), np.zeros(shape=(n, 1)), np.zeros(shape=(n, 1))

        self.N = np.zeros(shape=(n, 1))

    def bogdanov_hebb(self):
        return kpnet.utils.zero_diagonal(self.N * self.N.T)

    def activation_function(self, x):
        return np.heaviside(x, 0)

    def process(self, S): 
        self.W = (self.x1 + self.x2) * self.W0
        self.N = self.activation_function(self.P - self.h)
        
        norm = np.sum(self.N) + 1
        c = 1
        if norm != 0:
            c = 1. / norm 
        self.P = (1 - self.alpha) * self.P + c * np.dot(self.W, self.N) - self.beta * self.N + S 
            
        self.x1 = (1 - self.A1) * self.x1 + self.B1 * self.N + self.C1 
        self.x2 = (1 - self.A2) * self.x2 - self.B2 * self.N + self.C2

        BH = self.bogdanov_hebb()
        self.W0 = (1 - self.mu) * self.W0 + self.nu * BH 

        return self.N


class KPNetworkDelayed(KPNetwork):
    def __init__(self, n, m, alpha=0.2, beta=1.0):
        super(KPNetworkDelayed, self).__init__(n, alpha, beta)
        self.m = m
        self.D = np.zeros(shape=(m, n))

    def bogdanov_hebb(self):
        BH = self.N * np.sum(self.D, axis=0, keepdims=True) 

        self.D = np.roll(self.D, self.m - 1, axis=0)
        self.D[self.m - 1] = self.N.reshape(self.n)

        return kpnet.utils.zero_diagonal(BH)


class KPNetworkNormal(KPNetwork):
    def __init__(self, n, alpha=0.2, beta=1.0, sigma=1.0):
        super(KPNetworkNormal, self).__init__(n, alpha, beta)
        self.sigma = sigma

    def activation_function(self, x):
        return norm.cdf(x, 0, self.sigma)


class KPNetworkNormalDelayed(KPNetworkDelayed):
    def __init__(self, n, m, alpha=0.2, beta=1.0, sigma=1.0):
        super(KPNetworkNormalDelayed, self).__init__(n, m, alpha, beta)
        self.sigma = sigma

    def activation_function(self, x):
        return norm.cdf(x, 0, self.sigma)


class KPNetworkSigmoid(KPNetwork):
    def __init__(self, n, alpha=0.2, beta=1.0, gamma=0.1, border=1.0):
        super(KPNetworkSigmoid, self).__init__(n, alpha, beta)
        self.gamma = gamma
        self.h     = border * np.ones(shape=(n, 1))

    def activation_function(self, x):
        return kpnet.utils.sigmoid(x, self.gamma)


class KPNetworkTanh(KPNetwork):
    def __init__(self, n, alpha=0.2, beta=1.0, gamma=0.1):
        super(KPNetworkTanh, self).__init__(n, alpha, beta)
        self.gamma = gamma

    def activation_function(self, x):
        return np.tanh(self.gamma * x)


class KPNetworkTanhDelayed(KPNetworkDelayed):
    def __init__(self, n, m, alpha=0.2, beta=1.0, gamma=0.1):
        super(KPNetworkTanhDelayed, self).__init__(n, m, alpha, beta)
        self.gamma = gamma

    def activation_function(self, x):
        return np.tanh(self.gamma * x)


class KPNetworkSigmoidDelayed(KPNetworkDelayed):
    def __init__(self, n, m, alpha=0.2, beta=1.0, gamma=0.1, border=1.0):
        super(KPNetworkSigmoidDelayed, self).__init__(n, m, alpha, beta)
        self.gamma = gamma
        self.h     = border * np.ones(shape=(n, 1))

    def activation_function(self, x):
        return kpnet.utils.sigmoid(x, self.gamma)
