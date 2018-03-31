class ComputationHandler(object):
    def __init__(self, time_interval):
        self.time_interval = time_interval

    def run(self, network, signal, callbacks):
        results = {}
        for step in range(self.time_interval):
            network.process(signal[step])

            for function in callbacks.keys():
                callbacks[function].compute(network, step)

        for function in callbacks.keys():
            results[function] = callbacks[function].result.squeeze()

        return results

