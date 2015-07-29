#!/usr/bin/env python
import time
import gzip
import cPickle

import numpy as np
import theano
import theano.tensor as T

import models
from utils import normalize

# CONF #
N_EXAMPLES = 60000
N_TRAIN = 40000
N_TEST= 10000

N_INPUTS = 800
N_OUTPUTS = 1

LAYERS = [N_INPUTS, 50, N_OUTPUTS]
ACT_LIST = ([T.nnet.sigmoid] * (len(LAYERS) - 1))

DATA_FILENAME = 'data/reg_gen_set.pkl.gz'
########

# For skewing weights. Tweak as wanted.
def skew(w):
    if w < -0.5:
        return w + 5
    elif w > 0.5:
        return w - 5
    return 0.
skew = np.vectorize(skew)

########

n_valid = N_EXAMPLES - N_TRAIN - N_TEST

rng = np.random.RandomState(int(time.time()))
model = models.MLP(rng, LAYERS, ACT_LIST, [0., 0.], 'float',
                lambda output: output[0])

for layer in model.layers:
    R = rng.normal(loc=0.0, scale=0.5, size=(layer.n_in, layer.n_nodes))
    W = normalize(skew(R), range_=(-1., 1.))
    layer.set_W(W)

# abs to "fold" distribution
inputs = np.abs(rng.normal(loc=10.0, scale=1.0, size=(N_EXAMPLES, N_INPUTS)),)
inputs[inputs > 9.] = 0.
# Normalizing to 0-1
inputs = normalize(inputs)
inputs = theano.shared(inputs, borrow=True)

inputs_floatX = T.cast(inputs, theano.config.floatX)

targets = theano.function(
                inputs=[],
                outputs=model.output(),
                givens={model.x: inputs_floatX})()

print "Generated dataset."

print "Min y:", np.min(targets)
print "Max y:", np.max(targets)
print "Mean y:", np.mean(targets)
print "Median y:", np.median(targets)
print "y var:", np.var(targets)
print "y std:", np.std(targets)

inputs = inputs.get_value(borrow=True)

train_set = (inputs[:N_TRAIN], targets[:N_TRAIN])
valid_set = (inputs[N_TRAIN:N_TRAIN+n_valid], targets[N_TRAIN:N_TRAIN+n_valid])
test_set = (inputs[N_TRAIN+n_valid:], targets[N_TRAIN+n_valid:])

pack = (train_set, valid_set, test_set)

datafile = gzip.open(DATA_FILENAME, 'wb+')
cPickle.dump(pack, datafile, protocol=cPickle.HIGHEST_PROTOCOL)

print "Dumped data to file:", DATA_FILENAME
