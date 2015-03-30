#!/usr/bin/env python
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', type=int, default=int(time.time()))
# General MLP
parser.add_argument('-i', '--input', type=int, default=28*28)
parser.add_argument('-o', '--output', type=int, default=10)
parser.add_argument('-l', '--layers', type=int, nargs='+', default=[500])
# Dropout
parser.add_argument('-p', '--dropout-p', type=float, nargs='+', default=[0.0, 0.0])
# Training
parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
parser.add_argument('-lrd', '--learning-rate-decay', type=float, default=None)
parser.add_argument('-e', '--epochs', type=int, default=1000)
# Regularization
parser.add_argument('-l1', '--l1-reg', type=float, default=0.0)
parser.add_argument('-l2', '--l2-reg', type=float, default=0.0)
parser.add_argument('-m', '--max-col-norm', type=float, default=None)
# Active learning
parser.add_argument('--active', dest='active', action='store_true')
parser.add_argument('--no-active', dest='active', action='store_false')
parser.set_defaults(active=True)
parser.add_argument('-ebc', '--epochs-between-copies', type=int, default=1)
parser.add_argument('-r', '--random-sampling', type=bool, default=False)
parser.add_argument('-b', '--baseline-n', type=int, default=None)
args = parser.parse_args()
print args

# Importing after arg parsing so that we can run --help without locking a GPU.
import numpy as np
import theano.tensor as T

import mnist
import mlp
import trainers

print "Loading dataset."
datasets = mnist.load_data('mnist.pkl.gz')

if args.baseline_n is not None:
    # Baseline the active approach.
    train_set, valid_set, test_set = datasets
    train_set = train_set[0][args.baseline_n:], train_set[1][args.baseline_n:]
    datasets = train_set, valid_set, test_set

start_time = time.clock()
rng = np.random.RandomState(args.seed)

print "Generating model."
model = mlp.MLP(rng, args.input, args.layers, args.output, args.dropout_p, [T.tanh])

def neg_log_cost_w_l1_l2(y, config):
    return model.neg_log_likelihood(y) \
        + config['l1_reg'] * model.L1() \
        + config['l2_reg'] * model.L2()

trainer_config = { 
    'batch_size': 20, 
    'initial_learning_rate': args.learning_rate,
    'learning_rate_decay': args.learning_rate_decay,
    'max_col_norm': args.max_col_norm,
    'random_sampling': args.random_sampling,
    'l1_reg': args.l1_reg,
    'l2_reg': args.l2_reg,
    'epochs_between_copies': args.epochs_between_copies,
    # Initialize labeled pool with 240 examples (like Nguyen & Smulders 2004).
    'n_boostrap_examples': 240
}

if args.active:
    trainer = trainers.ActiveBackpropTrainer(model, neg_log_cost_w_l1_l2, datasets, trainer_config)
else:
    trainer = trainers.BackpropTrainer(model, neg_log_cost_w_l1_l2, datasets, trainer_config)

print "Training."
best_validation_loss, test_score = trainer.train(args.epochs)
print(('Optimization complete. Best validation score: %f %%. '
       'Test performace %f %%.') %
      (best_validation_loss * 100., test_score * 100.))
end_time = time.clock()
print 'The code ran for %.2fm' % ((end_time - start_time) / 60.)
