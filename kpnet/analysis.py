import pywt
import scipy.stats
import numpy as np

def wdist(signal, wname):
    level = pywt.dwt_max_level(signal.shape[0], pywt.Wavelet(wname))
    C = pywt.wavedec(signal, wname, level=level)
    d = [np.mean(c ** 2) for c in C]
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