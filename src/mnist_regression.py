import numpy as np

import mnist

def load_data(dataset):
    def to_reg_set(set_tuple):
        target = np.zeros((set_tuple[1].shape[0], 10))
        for i in xrange(set_tuple[1].shape[0]):
            target[i][set_tuple[1][i]] = 1.

        return (set_tuple[0], target)
    
    return tuple(map(to_reg_set, mnist.load_data(dataset)))
