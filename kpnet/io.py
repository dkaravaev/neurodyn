import scipy.io 
import os.path

def tomat(foldername, results):
    prefix = ''
    for key in ['alpha', 'beta', 'gamma', 'n']:
        prefix += str(results[key]) + '+'
    prefix = os.path.join(foldername, prefix[:-1])
    scipy.io.savemat(prefix + '.mat', results)
    