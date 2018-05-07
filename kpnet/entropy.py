import pywt
import wavelets
import numpy as np
import scipy.stats


class WTDistribution(object):
    def __init__(self):
        self.value = None

    @property
    def entropy(self):
        return scipy.stats.entropy(self.value)
    
    @property
    def moda(self):
        return np.argmax(self.value)
    
    def divergence(self, other):
        return scipy.stats.entropy(self.value, other.value)
    
    def jensen(self, other):
        return .5 * (self.divergence(other) + other.divergence(self))


class DWTDistribution(WTDistribution):
    def __init__(self, signal, wavelet):
        super(DWTDistribution, self).__init__()
        level = pywt.dwt_max_level(signal.shape[0], pywt.Wavelet(wavelet).dec_len)
        dwt = pywt.wavedec(signal, wavelet, level=level)
        self.value = np.flip(np.asarray([np.sum(coeffs ** 2) for coeffs in dwt]), axis=-1)
        self.value /= np.sum(self.value)


class SWTDistribution(WTDistribution):
    def __init__(self, signal, wavelet):
        super(SWTDistribution, self).__init__()
        self.level = pywt.swt_max_level(signal.shape[0])
        self.swt = np.asarray([D ** 2  for _, D in pywt.swt(signal, wavelet, level=self.level)])
        self.value = np.flip(np.sum(self.swt, axis=1) / np.sum(self.swt), axis=-1)


class CWTDistribution(WTDistribution):
    # todo: optimize coi mapping
    def __init__(self, signal, wavelet):
        super(CWTDistribution, self).__init__()
        wa = wavelets.WaveletTransform(signal, wavelet=wavelet)
        coi = wa.wavelet.coi
        s = wa.scales
        t = wa.time
        T, S = np.meshgrid(t, s)
        inside_coi = (coi(S) < T) & (T < (T.max() - coi(S)))
        masked_power = np.ma.masked_where(~inside_coi, wa.wavelet_power)
        masked_power.set_fill_value(0)
        self.value = np.sum(masked_power.filled(), axis=1) / np.sum(masked_power.filled()) 

