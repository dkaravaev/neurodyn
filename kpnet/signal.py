import struct
import numpy as np


class Signal(object):
    def __init__(self, neurons):
        self.neurons = neurons

    def __getitem__(self, step):
        pass

class ZeroSignal(Signal):
    def __init__(self, neurons):
        super(ZeroSignal, self).__init__(neurons)

    def __getitem__(self, step):
        return np.zeros(shape=(self.neurons, 1))

class RandomUnitSignal(Signal):
    def __init__(self, neurons, amplitude=0.5):
        super(RandomUnitSignal, self).__init__(neurons)
        self.amplitude = amplitude

    def __getitem__(self, step):
        i = np.random.randint(0, self.neurons)
        x = np.zeros(shape=(self.neurons, 1))
        x[i, 0] = self.amplitude
        
        return x


class WhiteNoiseSignal(Signal):
    def __init__(self, neurons, mean=0, std=1):
        super(WhiteNoiseSignal, self).__init__(neurons)
        self.mean = mean
        self.std = std

    def __getitem__(self, step):
        return np.random.normal(loc=self.mean, scale=self.std, size=(self.neurons, 1))


class SwitchSignal(Signal):
    def __init__(self, timestamps, signals):
        super(SwitchSignal, self).__init__(0)
        self.timestamps = timestamps
        self.signals = signals

        self.timestamp_index = 0

    def __getitem__(self, step):
        if step <= self.timestamps[self.timestamp_index]:
            return self.signals[self.timestamp_index][step]
        else:
            if self.timestamp_index != len(self.timestamps) - 1: 
                self.timestamp_index += 1
                return self.signals[self.timestamp_index][step]
            else:
                print("Error! Returning zeros.")
                return np.zeros(shape=(neurons, 1))


    def __getitem__(self, step):
        if step <= self.timestamps[self.timestamp_index]:
            return self.signals[self.timestamp_index][step]
        else:
            if self.timestamp_index != len(self.timestamps) - 1: 
                self.timestamp_index += 1
                return self.signals[self.timestamp_index][step]
            else:
                print("Error! Returning zeros.")
                return np.zeros(shape=(neurons, 1))


class PatternSignal(Signal):
    def __init__(self, neurons, patterns):
        super(PatternSignal, self).__init__(neurons)
        self.patterns = patterns

    def __getitem__(self, step):
        return self.patterns[np.random.randint(0, len(self.patterns))].reshape((self.neurons, 1))
