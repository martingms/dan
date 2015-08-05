#!/usr/bin/env python
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', type=int, default=int(time.time()))
parser.add_argument('-d', '--dataset', type=str, default='mnist')
# General MLP
# TODO derive these from data dimensions.
#parser.add_argument('-i', '--input', type=int, default=28*28)
#parser.add_argument('-i', '--input', type=int, default=520)
#parser.add_argument('-o', '--output', type=int, default=10)
#parser.add_argument('-o', '--output', type=int, default=2)
#parser.add_argument('-l', '--layers', type=int, nargs='+', default=[1000, 1000, 1000])
parser.add_argument('-l', '--layers', type=int, nargs='+', default=[1000])
# Dropout
#parser.add_argument('-p', '--dropout-p', type=float, nargs='+', default=[0.2, 0.5, 0.5, 0.5])
parser.add_argument('-p', '--dropout-p', type=float, nargs='+', default=[0.2, 0.5])
# Training
parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
parser.add_argument('-lrd', '--learning-rate-decay', type=float, default=None)
parser.add_argument('-e', '--epochs', type=int, default=10000)
# Regularization
parser.add_argument('-l1', '--l1-reg', type=float, default=0.0)
parser.add_argument('-l2', '--l2-reg', type=float, default=0.0)
parser.add_argument('-m', '--max-col-norm', type=float, default=None)
# Active learning
parser.add_argument('--active', dest='active', action='store_true')
parser.add_argument('--no-active', dest='active', action='store_false')
parser.set_defaults(active=True)
parser.add_argument('-sl', '--selector', type=str, default="oe")
parser.add_argument('-ebc', '--epochs-between-copies', type=int, default=10)
parser.add_argument('-b', '--baseline-n', type=int, default=None)
parser.add_argument('-n', '--n-select', type=int, default=10)
parser.add_argument('-ns', '--n-samples', type=int, default=1)
# Other
parser.add_argument('--dbn', dest='dbn', action='store_true')
parser.add_argument('--no-dbn', dest='dbn', action='store_false')
parser.set_defaults(dbn=True)
parser.add_argument('-pp', '--pickle-pretraining-file', type=str, default=None)
parser.add_argument('-lp', '--load-pretraining-file', type=str, default=None)
args = parser.parse_args()
print args

# Importing after arg parsing so that we can run --help without locking a GPU.
import sys
import numpy as np
import theano.tensor as T
import cPickle

import mnist
import mnist_regression
import ujindoor
import gen_reg_set_load

import models
import trainers
import activeselectors
import errorfuncs

if args.active:
    if args.selector == "oe":
        print "Using output entropy selector."
        selector = activeselectors.OutputEntropy
    elif args.selector == "sve":
        print "Using soft vote entropy selector."
        selector = activeselectors.SoftVoteEntropy
    elif args.selector == "kld":
        print "Using kullback leibler divergence selector."
        selector = activeselectors.KullbackLeiblerDivergence
    elif args.selector == "rand":
        print "Using random selector."
        selector = activeselectors.Random
    elif args.selector == "svar":
        print "Using samle variance selector."
        selector = activeselectors.SampleVariance
    elif args.selector == "jsvar":
        print "Using josh's pointwise sample variance selector."
        selector = activeselectors.PointwiseSampleVariance
    else:
        print "No such selector!"
        sys.exit(1)
else:
    selector = None

print "Loading dataset:", args.dataset
if args.dataset == 'mnist':
    datasets = mnist.load_data('mnist.pkl.gz')
    inputs = 28*28
    outputs = 10
elif args.dataset == 'mnistreg':
    datasets = mnist_regression.load_data('mnist.pkl.gz')
    inputs = 28*28
    outputs = 10
elif args.dataset == 'ujindoor':
    datasets = ujindoor.load_data(
        'data/UJIndoorLoc/trainingData_shuffled.csv',
        'data/UJIndoorLoc/validationData_shuffled.csv'
    )
    inputs = 520
    outputs = 2
elif args.dataset == 'gen':
    datasets = gen_reg_set_load.load_data('data/reg_gen_set.pkl.gz')
    inputs = 800
    outputs = 1
else:
    print "No such dataset:", args.dataset
    sys.exit(1)

if args.baseline_n is not None:
    # Baseline the active approach.
    print "Training set limited to first", args.baseline_n, "examples."
    train_set, valid_set, test_set = datasets
    train_set = train_set[0][:args.baseline_n], train_set[1][:args.baseline_n]
    datasets = train_set, valid_set, test_set

start_time = time.clock()
rng = np.random.RandomState(args.seed)

if not args.load_pretraining_file:
    print "Generating model:",
    layers = [inputs] + args.layers + [outputs]

    if args.dbn:
        activation_list = [T.nnet.sigmoid] * len(args.layers)
    else:
        activation_list = [T.tanh] * len(args.layers)

    if args.dataset == 'ujindoor':
        activation_list = activation_list + [lambda x: x]
        datatype = 'float'
        output_func = lambda output: output
    elif args.dataset == 'mnistreg' or args.dataset == 'gen':
        activation_list = activation_list + [T.nnet.sigmoid]
        datatype = 'float'
        output_func = lambda output: output
    elif args.dataset == 'mnist':
        activation_list = activation_list + [T.nnet.softmax]
        datatype = 'int'
        output_func = lambda output: T.argmax(output, axis=1)

    if args.dbn:
        print "DBN."
        model = models.DBN(rng, layers, activation_list, args.dropout_p,
                        datatype, output_func)
    else:
        print "MLP."
        model = models.MLP(rng, layers, activation_list, args.dropout_p,
                        datatype, output_func)
else:
    print "Loading model from pickled file."
    f = file(args.load_pretraining_file, 'rb')
    model = cPickle.load(f)
    f.close()

trainer_config = { 
    'batch_size': 10, 
    'initial_learning_rate': args.learning_rate,
    'learning_rate_decay': args.learning_rate_decay,
    'max_col_norm': args.max_col_norm,
    'l1_reg': args.l1_reg,
    'l2_reg': args.l2_reg,
    'active_selector': selector,
    'n_samples': args.n_samples,
    'n_select': args.n_select,
    'epochs_between_copies': args.epochs_between_copies,
    # Initialize labeled pool in active learning with 240 examples (like Nguyen
    # & Smulders 2004).
    'n_boostrap_examples': 240,
    # DBN
    'pretrain_lr': 0.01,
    'k': 1
}

# Cost functions used for training.
# TODO: Move this within the lib?
if args.dataset == 'mnist':
    def neg_log_cost_w_l1_l2(y, config):
        return model.neg_log_likelihood(y) \
            + config['l1_reg'] * model.L1() \
            + config['l2_reg'] * model.L2()
    cost_func = neg_log_cost_w_l1_l2
    error_func = errorfuncs.meanerrors

elif args.dataset == 'ujindoor' or args.dataset == 'gen' or args.dataset == 'mnistreg':
    def rmse(y, config):
        return model.rmse(y)
    cost_func = rmse
    error_func = errorfuncs.rmse

if args.dbn and not args.load_pretraining_file:
    print "Initializing pretrainer."
    pretrainer = trainers.DBNTrainer(model, cost_func, error_func, datasets, trainer_config)
    print "Pretraining."
    pretrainer.pre_train(100)

if args.dbn and args.pickle_pretraining_file:
    print "Pickling pretrained model."
    f = file(args.pickle_pretraining_file, 'wb')
    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

if args.active:
    print "Using active trainer."
    trainer = trainers.ActiveBackpropTrainer(model, cost_func, error_func, datasets, trainer_config)
elif args.dbn and not args.load_pretraining_file:
    print "Using DBN trainer." # Simply inherited from BackpropTrainer
    trainer = pretrainer
else:
    print "Using normal backprop trainer."
    trainer = trainers.BackpropTrainer(model, cost_func, error_func, datasets, trainer_config)

print "Training."
best_validation_loss, test_score = trainer.train(args.epochs)
print(('Optimization complete. Best validation score: %f . '
       'Test performace %f .') %
      (best_validation_loss, test_score))
end_time = time.clock()
print 'The code ran for %.2fm' % ((end_time - start_time) / 60.)
