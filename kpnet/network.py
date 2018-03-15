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

    def learning_rule(self):
        BH = kpnet.utils.zero_diagonal(self.N * self.N.T)
        return (1 - self.mu) * self.W0 + self.nu * BH

    def activation_function(self, x):
        return np.heaviside(x, 0)

    def process(self, S): 
        self.W = (self.x1 + self.x2) * self.W0
        self.N = self.activation_function(self.P - self.h)
        
        norm = np.sum(self.N) + 1
        c = 1
        if norm != 0:
            c = 1. / norm 
        self.P = (1 - self.alpha) * self.P + c * np.dot(self.W, self.N) - self.beta * self.N + S + np.random.normal(0, 1)
 
            
        self.x1 = (1 - self.A1) * self.x1 + self.B1 * self.N + self.C1 
        self.x2 = (1 - self.A2) * self.x2 - self.B2 * self.N + self.C2

        self.W0 = self.learning_rule() 

        return self.N

class KPNetworkDelayed(KPNetwork):
    def __init__(self, n, m, alpha=0.2, beta=1.0):
        super(KPNetworkDelayed, self).__init__(n, alpha, beta)
        self.m = m
        self.D = np.zeros(shape=(m, n))

    def learning_rule(self):
        BH = self.N * np.sum(self.D, axis=0, keepdims=True)
        self.D = np.roll(self.D, self.m - 1, axis=0)
        self.D[self.m - 1] = self.N.reshape(self.n)

        return kpnet.utils.zero_diagonal((1 - self.mu) * self.W0 + self.nu * BH)

class KPNetworkOja(KPNetwork):
    def __init__(self, n, alpha=0.2, beta=1.0, gamma=0.1):
        super(KPNetworkOja, self).__init__(n, alpha, beta)
        self.gamma = gamma

    def learning_rule(self):
        return self.gamma * (self.N * self.N.T + self.N * self.W0)

class KPNetworkCov(KPNetwork):
    def __init__(self, n, alpha=0.2, beta=1.0):
        super(KPNetworkCov, self).__init__(n, alpha, beta)
        self.means = self.N
        self.step = 0

    def learning_rule(self):
        self.step += 1
        self.means = ((self.step - 1) * self.means + self.N) / self.step 
        shifted = self.N - self.means
        return (1 - self.mu) * self.W0 + self.nu * kpnet.utils.zero_diagonal(shifted * shifted.T)

class KPNetworkCovTanh(KPNetwork):
    def __init__(self, n, alpha=0.2, beta=1.0, gamma=0.1, epsilon=1.0):
        super(KPNetworkCovTanh, self).__init__(n, alpha, beta)
        self.h = 2 * np.ones(shape=(n, 1))
        self.means = self.N
        self.step = 0
        self.gamma = gamma
        self.epsilon = epsilon

    def learning_rule(self):
        self.step += 1
        self.means = ((self.step - 1) * self.means + self.N) / self.step 
        shifted = self.N - self.means
        return self.W0 + kpnet.utils.zero_diagonal(self.gamma * shifted * shifted.T)

    def activation_function(self, x):
        return (1 + np.tanh(self.epsilon * x)) / 2.


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
        self.h = 1 + np.zeros(shape=(n, 1))

    def activation_function(self, x):
        return (1 + np.tanh(self.gamma * x)) / 2

class KPNetworkTanhDelayed(KPNetworkDelayed):
    def __init__(self, n, m, alpha=0.2, beta=1.0, gamma=0.1):
        super(KPNetworkTanhDelayed, self).__init__(n, m, alpha, beta)
        self.h = 1 * np.ones(shape=(n, 1))
        self.gamma = gamma

    def activation_function(self, x):
        return  (1 + np.tanh(self.gamma * x)) / 2.


class KPNetworkSigmoidDelayed(KPNetworkDelayed):
    def __init__(self, n, m, alpha=0.2, beta=1.0, gamma=0.1, border=1.0):
        super(KPNetworkSigmoidDelayed, self).__init__(n, m, alpha, beta)
        self.gamma = gamma
        self.h     = border * np.ones(shape=(n, 1))

    def activation_function(self, x):
        return kpnet.utils.sigmoid(x, self.gamma)
