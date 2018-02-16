import pywt
import wavelets
import scipy.stats
import numpy as np


def wdist_cwt(signal):
    P = wavelets.WaveletAnalysis(signal, wavelet=wavelets.Ricker()).wavelet_power
    return np.sum(P, axis=0) / np.sum(P)

def wentropy_cwt(signal):
    return scipy.stats.entropy(wdist_cwt(signal))

def wentropy_cwtd(signal, step):
    we = np.zeros(signal.shape[0])
    for i in range(int(we.shape[0] / step)):
        we[i * step : (i + 1) * step] = wentropy_cwt(signal[i * step : (i + 1) * step])
    return we

def wdist(signal, wname):
    level = pywt.dwt_max_level(signal.shape[0], pywt.Wavelet(wname))
    C = pywt.wavedec(signal, wname, level=level)
    d = [np.mean(C[i] ** 2) for i in range(1, level + 1)]
    return d / np.sum(d)

def wentropy(signal, wname):
    return scipy.stats.entropy(wdist(signal, wname))

def wdivergence(s1, s2, wname):
    return scipy.stats.entropy(wdist(s1, wname), wdist(s2, wname))

def wentropy_d(signal, step, wname):
    we = np.zeros(signal.shape[0])
    for i in range(int(we.shape[0] / step)):
        we[i * step : (i + 1) * step] = wentropy(signal[i * step : (i + 1) * step], wname)
    return we

def wentropy_sd(signal, step, wname):
    we = np.zeros(signal.shape[0])
    for i in range(int(we.shape[0] - step)):
        we[i] = wentropy(signal[i : i + step], wname)
    return we

def wdivergence_d(s1, s2, step, wname):
    we_l = np.zeros(s1.shape[0])
    #we_r = np.zeros(s2.shape[0])
    for i in range(int(we_l.shape[0] / step)):
        we_l[i * step : (i + 1) * step] = wdivergence(s1[i * step : (i + 1) * step], s2[i * step : (i + 1) * step], wname)
        #we_r[i * step : (i + 1) * step] = wdivergence(s2[i * step : (i + 1) * step], s1[i * step : (i + 1) * step], wname)
    return we_l#, we_r

def wdivergence_sd(s1, s2, step, wname):
    we_l = np.zeros(s1.shape[0])
    we_r = np.zeros(s2.shape[0])
    for i in range(int(we_l.shape[0] - step)):
        we_l[i : i + step] = wdivergence(s1[i : i + step], s2[i : i + step], wname)
        we_r[i : i + step] = wdivergence(s2[i : i + step], s1[i : i + step], wname)
    return we_l, we_r