#!/usr/bin/env python
import time
import gzip
import cPickle

import numpy as np
import theano
import theano.tensor as T

import models

# CONF #
N_EXAMPLES = 60000
N_TRAIN = 40000
N_TEST= 10000

N_INPUTS = 800
N_OUTPUTS = 1

LAYERS = [N_INPUTS, 1000, 1000, N_OUTPUTS]
ACT_LIST = ([T.nnet.sigmoid] * (len(LAYERS) - 2)) + [lambda x: x]

DATA_FILENAME = 'data/reg_gen_set.pkl.gz'
########

n_valid = N_EXAMPLES - N_TRAIN - N_TEST
dropout_ps = [0.0, 0.0, 0.0]

rng = np.random.RandomState(int(time.time()))
model = models.MLP(rng, LAYERS, ACT_LIST, dropout_ps, 'float',
                lambda output: output[0])

for layer in model.layers:
    W = rng.normal(loc=0.0, scale=20.0, size=(layer.n_in, layer.n_nodes))
    layer.set_W(W)

inputs = theano.shared(
                rng.normal(scale=10.0, size=(N_EXAMPLES, N_INPUTS)),
                borrow=True)

targets = theano.function(
                inputs=[],
                outputs=model.output(),
                givens={model.x: inputs})()

print "Generated dataset."

print "Min y:", np.min(targets)
print "Max y:", np.max(targets)
print "Mean y:", np.mean(targets)
print "Median y:", np.median(targets)
print "y var:", np.var(targets)
print "y std:", np.std(targets)

train_set = (inputs[:N_TRAIN], targets[:N_TRAIN])
valid_set = (inputs[N_TRAIN:N_TRAIN+n_valid], targets[N_TRAIN:N_TRAIN+n_valid])
test_set = (inputs[N_TRAIN+n_valid:], targets[N_TRAIN+n_valid:])

pack = (train_set, valid_set, test_set)

datafile = gzip.open(DATA_FILENAME, 'wb+')
cPickle.dump(pack, datafile, protocol=cPickle.HIGHEST_PROTOCOL)

print "Dumped data to file:", DATA_FILENAME
