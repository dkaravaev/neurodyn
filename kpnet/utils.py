import numpy as np

from wavelets import WaveletTransform

def sigmoid(x, gamma):
    return 1 / (1 + np.exp(-gamma * x))

def zero_diagonal(matrix):
    return matrix - np.diag(np.diag(matrix))

def moving_average(a, n=500) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_cwt(ax, signal, wavelet):
    wa = WaveletAnalysis(signal, wavelet=wavelet)
    Time, Scale = np.meshgrid(wa.time, wa.scales)
    CS = ax.contourf(Time, Scale, wa.wavelet_power, 100)

    ax.set_yscale('log')
    ax.grid(True)

    coi_time, coi_scale = wa.coi
    ax.fill_between(x=coi_time,
                    y1=coi_scale,
                    y2=wa.scales.max(),
                    color='gray',
                    alpha=0.3)

    ax.set_xlim(wa.time.min(), wa.time.max())
    return CS