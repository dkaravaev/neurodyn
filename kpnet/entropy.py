import pywt
import wavelets
import scipy.stats
import numpy as np


class WaveletEntropy:
    @staticmethod
    def distribution(matrix, axis=1):
        d = np.mean(matrix, axis=axis)
        return d / np.sum(d)
    
    @staticmethod
    def weighted_distribution(matrix, axis=1):
        d = np.flip(np.arange(1, matrix.shape[0] + 1), axis=-1) * np.sum(matrix, axis=axis)
        return d / np.sum(d)
    
    @staticmethod
    def unnormalized_distribution(matrix, axis=1):
        d = np.mean(matrix, axis=axis)
        ind = np.argwhere(d == 0)
        d = np.delete(d, ind)
        return d
    
    @staticmethod
    def raw(ps, axis=1):
        return -np.sum(ps * np.log(ps))
    
    @staticmethod
    def shannon(ps):
        return scipy.stats.entropy(ps)
    
    @staticmethod
    def divergence(ps, qs):
        return scipy.stats.entropy(ps, qs)
    
    @staticmethod
    def jensen(ps, qs):
        return .5 * (WaveletEntropy.divergence(ps, qs) + WaveletEntropy.divergence(qs, ps))
    
    def __init__(self, power_matrix):
        self.power_matrix  = power_matrix
    
    @property
    def overall(self):
        return self.entropy(self.distribution(self.power_matrix, axis=1))

    def sliced(self, step, fd, fe):
        result = np.zeros(self.power_matrix.shape[1])
        for i in range(int(result.shape[0] / step)):
            ps = fd(self.power_matrix[:, i * step : (i + 1) * step])
            result[i * step : (i + 1) * step] = fe(ps)
        return result
    
    @staticmethod
    def compare(ep, eq, fd, fe):
        ps = fd(ep.power_matrix)
        qs = fd(eq.power_matrix)
        return fe(ps, qs)
    
    @staticmethod
    def compare_sliced(step, ep, eq, fd, fe):
        result = np.zeros(ep.power_matrix.shape[1])
        for i in range(int(result.shape[0] / step)):
            ps = fd(ep.power_matrix[:, i * step : (i + 1) * step])
            qs = fd(eq.power_matrix[:, i * step : (i + 1) * step])
            result[i * step : (i + 1) * step] = fe(ps, qs)
        return result
        
    
class EntropyCWT(WaveletEntropy):
    def __init__(self, signal, wavelet):
        wa = wavelets.WaveletTransform(signal, wavelet=wavelet)
        coi = wa.wavelet.coi
        s = wa.scales
        t = wa.time
        T, S = np.meshgrid(t, s)
        inside_coi = (coi(S) < T) & (T < (T.max() - coi(S)))
        masked_power = np.ma.masked_where(~inside_coi, wa.wavelet_power)
        masked_power.set_fill_value(0)
        super(EntropyCWT, self).__init__(masked_power.filled())


class EntropySWT(WaveletEntropy):
    def __init__(self, signal, wavelet):
        level = pywt.swt_max_level(signal.shape[0])
        power_matrix = np.asarray([d ** 2 for _, d in pywt.swt(signal, wavelet, level=level)])
        super(EntropySWT, self).__init__(power_matrix)


class EntropyDWT(WaveletEntropy):
    def __init__(self, signal, wavelet):
        level = pywt.dwt_max_level(signal.shape[0])
        power_matrix = np.asarray([d ** 2 for d in pywt.dwt(signal, wavelet, level=level)[1:]])
        super(EntropySWT, self).__init__(power_matrix)