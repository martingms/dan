import time
import math
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

from mnist import load_data, shared_dataset


class Layer(object):
    """A generic perceptron layer"""
    def __init__(self, input, n_in, n_nodes, W=None, b=None,
            activation=lambda x: x):
        if W == None:
            W = np.zeros((n_in, n_nodes), dtype=theano.config.floatX)
            W = theano.shared(value=W, name='W', borrow=True)
        if b == None:
            b = np.zeros((n_nodes,), dtype=theano.config.floatX)
            b = theano.shared(value=b, name='b', borrow=True)

        self.W = W
        self.b = b

        self.output = activation(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]

    @staticmethod
    def generate_W(rng, n_in, n_nodes):
        return theano.shared(value=np.asarray(
                      # Numbers from
                      # Y. Bengio, X. Glorot, Understanding the difficulty of
                      # training deep feedforward neuralnetworks, AISTATS 2010
                      rng.uniform(
                          low=-np.sqrt(6. / (n_in + n_nodes)),
                          high=np.sqrt(6. / (n_in + n_nodes)),
                          size=(n_in, n_nodes)
                      ),
                      dtype=theano.config.floatX
                  ),
                  name='W', borrow=True)

class DropoutLayer(Layer):
    """
    Perceptron layer that implements the dropout regularization method described
    in N. Srivastava, G. Hinton, et al., Dropout: A simple way to prevent neural
    networks from overfitting (JMLR 2014).
    """
    def __init__(self, srng, input, n_in, n_nodes, W=None, b=None,
            activation=lambda x: x, dropout_rate=0.5):
        dropout_mask = srng.binomial(n=1, p=1-dropout_rate, size=input.shape)

        # Keeps stuff on GPU
        input *= T.cast(dropout_mask, theano.config.floatX)

        super(DropoutLayer, self).__init__(input, n_in, n_nodes, W, b,
                activation)

class MLP(object):
    """TODO: Write docstring"""
    def __init__(self, rng, n_in, n_hidden_list, n_out, dropout_rate_list,
            activation_list):
        assert len(n_hidden_list) + 1 == len(dropout_rate_list)
        self.dropout = max(dropout_rate_list) > 0.0
        assert len(n_hidden_list) == len(activation_list)

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        ### Wire up network
        self.layers = []
        self.dropout_layers = []

        # Main rng used to seed shared rng. This is probably the easiest way to get determinism.
        self.rng = rng
        srng = T.shared_randomstreams.RandomStreams(self.rng.randint(2147483647))

        # Hidden layers
        dropout_input = self.x
        input = self.x
        for n_layer, dropout_rate, activation_func in zip(n_hidden_list,
                dropout_rate_list, activation_list):
            dropout_layer = DropoutLayer(
                srng=srng,
                input=dropout_input,
                n_in=n_in,
                n_nodes=n_layer,
                W=Layer.generate_W(rng, n_in, n_layer),
                activation=activation_func,
                dropout_rate=dropout_rate
            )
            dropout_input = dropout_layer.output
            self.dropout_layers.append(dropout_layer)
            layer = Layer(
                input=input,
                n_in=n_in,
                n_nodes=n_layer,
                # Scaling based on dropout.
                W=dropout_layer.W * (1-dropout_rate),
                b=dropout_layer.b,
                activation=activation_func
            )
            self.layers.append(layer)
            input = layer.output
            n_in = n_layer

        # Softmax output layer
        self.dropout_layers.append(DropoutLayer(
            srng=srng,
            input=dropout_input,
            n_in=n_in,
            n_nodes=n_out,
            W=Layer.generate_W(rng, n_in, n_out),
            activation=T.nnet.softmax,
            dropout_rate=dropout_rate_list[-1]
        ))
        self.layers.append(Layer(
            input=input,
            n_in=n_in,
            n_nodes=n_out,
            W=self.dropout_layers[-1].W * (1-dropout_rate_list[-1]),
            b=self.dropout_layers[-1].b * (1-dropout_rate_list[-1]),
            activation=T.nnet.softmax
        ))

        self.y_pred = T.argmax(self.layers[-1].output, axis=1)

        self.params = [param for layer in self.dropout_layers
        #self.params = [param for layer in self.layers
                             for param in layer.params]

    def neg_log_likelihood(self, y):
        if not self.dropout:
            return -T.mean(T.log(self.layers[-1].output)[T.arange(y.shape[0]), y])
        return -T.mean(T.log(self.dropout_layers[-1].output)[T.arange(y.shape[0]), y])

    def errors(self):
        if self.y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', self.y.type, 'y_pred', self.y_pred.type)
            )
        if not self.y.dtype.startswith('int'):
            raise NotImplementedError()
        return T.mean(T.neq(self.y_pred, self.y))

    def output_entropy(self):
        output = self.layers[-1].output
        return -T.sum(output * T.log(output), axis=1)

    def L1(self):
        return sum([abs(layer.W).sum() for layer in self.layers])

    def L2(self):
        return sum([(layer.W ** 2).sum() for layer in self.layers])

class BackpropTrainer(object):
    def __init__(self, model, cost, datasets, config):
        self.model = model
        self.config = config
        self.cost = cost
        self._init_datasets(datasets)
        self._init_theano_functions()

    def _init_datasets(self, datasets):
        train_set, valid_set, test_set = datasets
        self.train_set_x, _, self.train_set_y = shared_dataset(train_set)
        self.valid_set_x, _, self.valid_set_y = shared_dataset(valid_set)
        self.test_set_x, _, self.test_set_y = shared_dataset(test_set)
        self.n_train_batches = \
                train_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = \
                valid_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_test_batches = \
                test_set_x.get_value(borrow=True).shape[0] / batch_size

    def _init_theano_functions(self):
        # Common variables
        self.start = T.lscalar()
        self.stop = T.lscalar()

        self.learning_rate = theano.shared(
            np.cast[theano.config.floatX](self.config['initial_learning_rate'])
        )

        # Training function
        gparams = [T.grad(self.cost(self.model.y, self.config), param)
                   for param in self.model.params]
        train_updates = OrderedDict()
        for param, gparam in zip(self.model.params, gparams):
            train_updates[param] = param - self.learning_rate * gparam
        # Max-norm regularization
        max_col_norm = self.config['max_col_norm']
        if max_col_norm is not None:
            for param, stepped_param in train_updates.iteritems():
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(max_col_norm))
                scale = desired_norms / (1e-7 + col_norms)
                train_updates[param] = stepped_param * scale

        self.train_func = theano.function(
            inputs=[self.start, self.stop],
            outputs=self.cost(self.model.y, self.config),
            updates=train_updates,
            givens={
                self.model.x: self.train_set_x[self.start:self.stop],
                self.model.y: self.train_set_y[self.start:self.stop]
            }
        )

        # Learning rate decay function
        if self.config['learning_rate_decay'] is not None:
            self.learning_rate_update_func = theano.function(
                inputs=[],
                updates=(self.learning_rate,
                    self.learning_rate * self.config['learning_rate_decay'])
            )

        # Validation function
        self.validate_func = theano.function(
            inputs=[self.start, self.stop],
            outputs=self.model.errors(),
            givens={
                self.model.x: self.valid_set_x[self.start:self.stop],
                self.model.y: self.valid_set_y[self.start:self.stop]
            }
        )

        # Test function
        self.test_func = theano.function(
            inputs=[self.start, self.stop],
            outputs=self.model.errors(),
            givens={
                self.model.x: self.test_set_x[self.start:self.stop],
                self.model.y: self.test_set_y[self.start:self.stop]
            }
        )

    def _calc_train_batch_range(self, bindex):
        return self._calc_test_and_valid_batch_range(bindex)

    def _calc_test_and_valid_batch_range(self, bindex):
        return (bindex * self.config['batch_size'],
                (bindex + 1) * self.config['batch_size'])

    def _epoch(self):
        avg_costs = [self.train_func(*self._calc_train_batch_range(i))
                     for i in xrange(self.n_train_batches)]
        return np.mean(avg_costs)

    def _validate(self):
        # Compute zero-one loss on validation set.
        validation_losses = \
                [self.validate_func(*self._calc_test_and_valid_batch_range(i))
                 for i in xrange(self.n_valid_batches)]
        return np.mean(validation_losses)

    def _test(self):
         test_losses = \
                 [self.test_func(*self._calc_test_and_valid_batch_range(i))
                  for i in xrange(self.n_test_batches)]
         return np.mean(test_losses)

    def train(self, n_epochs=1000, cur_epoch=0):
        best_validation_loss = np.inf

        done_looping = False
        epoch = 0
        while (epoch < n_epochs and (not done_looping)):
            epoch += 1
            cur_epoch += 1
            avg_costs = self._epoch()
            validation_loss = self._validate()

            marker = ""
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                marker = "*"

            print(
                'epoch %i, avg training cost %f, validation error %f %%, learning_rate %f %s' %
                (
                    cur_epoch,
                    avg_costs,
                    validation_loss * 100.,
                    self.learning_rate.get_value(borrow=True),
                    marker
                )
            )

            if self.config['learning_rate_decay'] is not None:
                self.learning_rate_update_func()

        test_score = self._test()
        print "    epoch %i, test score %f %%" % (cur_epoch, test_score * 100.)

        return best_validation_loss, test_score

class ActiveBackpropTrainer(BackpropTrainer):
    def _init_datasets(self, datasets):
        train_set, valid_set, test_set = datasets
        # Split training set into labeled and unlabeled sets.
        # Initialize labeled pool with 240 examples (like Nguyen & Smulders 2004).
        train_set_x, train_set_y = train_set[0][:240], train_set[1][:240]
        # Pad with zeros so we don't have to resize when adding new examples to the pool.
        # How much to pad can be set to the max number of examples we want to add.
        # Erring on the side of padding too much for now.
        train_set_x = np.pad(train_set_x,
                ((0,len(train_set[0])-len(train_set_x)), (0,0)),
                mode='constant')
        train_set_y = np.pad(train_set_y,
                (0,len(train_set[1])-len(train_set_y)),
                mode='constant')
        self.train_set_x, self.train_set_y_float, self.train_set_y = \
                shared_dataset((train_set_x, train_set_y))

        # TODO/FIXME: Too long line...
        self.unlabeled_set_x, self.unlabeled_set_y_float, self.unlabeled_set_y = \
                shared_dataset((train_set[0][240:], train_set[1][240:]))

        self.train_set_ptr = 240
        self.unlabeled_set_ptr = len(train_set[0][240:]) - 1

        self.n_train_batches = \
                int(math.ceil(
                    self.train_set_ptr / float(self.config['batch_size'])))
        self.n_unlabeled_batches = \
                self.unlabeled_set_x.get_value(borrow=True).shape[0] \
                    / self.config['batch_size']

        # Validation and test sets are set just like in BackpropTrainer.
        self.valid_set_x, _, self.valid_set_y = shared_dataset(valid_set)
        self.test_set_x, _, self.test_set_y = shared_dataset(test_set)
        self.n_valid_batches = \
                self.valid_set_x.get_value(borrow=True).shape[0] \
                    / self.config['batch_size']
        self.n_test_batches = \
                self.test_set_x.get_value(borrow=True).shape[0] \
                    / self.config['batch_size']

    def _init_theano_functions(self):
        # Init all the common functions and variables.
        super(ActiveBackpropTrainer, self)._init_theano_functions()

        # Init the active learning-specific stuff.
        # Entropy function
        self.entropy_func = theano.function(
            inputs=[self.start, self.stop],
            outputs=self.model.output_entropy(),
            givens={
                self.model.x: self.unlabeled_set_x[self.start:self.stop],
            }
        )

        # Copy from unlabeled to training set function
        # Warning: Part of a terrible hack to avoid expensive resizing of matrices.
        # TODO/FIXME: Make this a OrderedDict
        idx = T.lscalar()
        copy_updates = [
            # Copy value at idx in unlabeled set to first free spot in training set.
            (self.train_set_x,
                T.set_subtensor(
                    self.train_set_x[self.train_set_ptr],
                    self.unlabeled_set_x[idx])),
            (self.train_set_y_float,
                T.set_subtensor(
                    self.train_set_y_float[self.train_set_ptr],
                    self.unlabeled_set_y_float[idx])),
            # Delete idx from unlabeled set by swapping in bottom and decreasing pointer.
            (self.unlabeled_set_x,
                T.set_subtensor(
                    self.unlabeled_set_x[idx],
                    self.unlabeled_set_x[self.unlabeled_set_ptr])),
            (self.unlabeled_set_y_float,
                T.set_subtensor(
                    self.unlabeled_set_y_float[idx],
                    self.unlabeled_set_y_float[self.unlabeled_set_ptr]))
        ]
        # Warning: This is wrapped in `_copy_to_train_set(idx)`.
        self.copy_to_train_set_func = theano.function(
            inputs=[idx],
            updates=copy_updates
        )

    def _copy_to_train_set(self, idx):
        """ Wrapper around `copy_to_train_set_func` to also set set-pointers
        correctly. As these are not SharedVariables, they can't be updated
        within a theano.function."""
        self.copy_to_train_set_func(idx)
        self.train_set_ptr += 1
        self.unlabeled_set_ptr -= 1
    
    def _calc_train_batch_range(self, bindex):
        start, stop = super(ActiveBackpropTrainer, self)._calc_train_batch_range(bindex)
        if stop > self.train_set_ptr:
            stop = self.train_set_ptr

        return start, stop
    
    def _calc_unlabeled_batch_range(self, bindex):
        start, stop = super(ActiveBackpropTrainer, self)._calc_train_batch_range(bindex)
        if stop > self.unlabeled_set_ptr + 1:
            stop = self.unlabeled_set_ptr + 1

        return start, stop

    def train(self, epochs, epochs_between_copies=1):
        # TODO: Implement copying more than one example.
        best_validation_loss = np.inf
        test_score = 0.
        total_epoch = 0
        for i in xrange(epochs/epochs_between_copies):
            validation_loss, test_score = super(ActiveBackpropTrainer,
                    self).train(epochs_between_copies, total_epoch)
            total_epoch += epochs_between_copies
            if validation_loss > best_validation_loss:
                best_validation_loss = validation_loss

            if not self.config['random_sampling']:
                # TODO/FIXME: Should probably reuse this buffer.
                entropies = np.empty(
                        (self.n_unlabeled_batches, self.config['batch_size']),
                        dtype=theano.config.floatX
                )

                for bindex in xrange(self.n_unlabeled_batches):
                    ent = self.entropy_func(
                            *self._calc_unlabeled_batch_range(bindex))
                    # The last batch can have an uneven size. In that case, we
                    # pad with zeros, since they don't mess up our results with
                    # np.argmax.
                    if len(ent) != 20:
                        ent = np.pad(ent, (0, 20-len(ent)), mode='constant')
                    entropies[i] = ent

                idx = np.argmax(entropies)
            else:
                idx = self.rng.randint(set_ptrs['unlabeled'])

            # Copy that example to training set and delete from unlabeled set.
            self._copy_to_train_set(idx)
            self.n_unlabeled_batches = \
                    int(math.ceil(
                        self.unlabeled_set_ptr \
                                / float(self.config['batch_size'])))
            self.n_train_batches = \
                    int(math.ceil(
                        self.train_set_ptr / float(self.config['batch_size'])))

        return best_validation_loss, test_score

if __name__ == '__main__':
    start_time = time.clock()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=int, default=28*28)
    parser.add_argument('-o', '--output', type=int, default=10)
    parser.add_argument('-l', '--layers', type=int, nargs='+', default=[500])
    parser.add_argument('-p', '--dropout-p', type=float, nargs='+', default=[0.0, 0.0])
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-lrd', '--learning-rate-decay', type=float, default=None)
    parser.add_argument('-s', '--seed', type=int, default=int(time.time()))
    parser.add_argument('-e', '--epochs', type=int, default=1000)
    parser.add_argument('-l1', '--l1-reg', type=float, default=0.0)
    parser.add_argument('-l2', '--l2-reg', type=float, default=0.0)
    parser.add_argument('-m', '--max-col-norm', type=float, default=None)
    parser.add_argument('-a', '--active', type=bool, default=True)
    parser.add_argument('-r', '--random-sampling', type=bool, default=False)
    args = parser.parse_args()
    print args

    print "Loading dataset."
    datasets = load_data('mnist.pkl.gz')

    rng = np.random.RandomState(args.seed)

    print "Generating model."
    mlp = MLP(rng, args.input, args.layers, args.output, args.dropout_p, [T.tanh])

    def neg_log_cost_w_l1_l2(y, config):
        return mlp.neg_log_likelihood(y) \
            + config['l1_reg'] * mlp.L1() \
            + config['l2_reg'] * mlp.L2()

    trainer_config = {
        'batch_size': 20,
        'initial_learning_rate': args.learning_rate,
        'learning_rate_decay': args.learning_rate_decay,
        'max_col_norm': args.max_col_norm,
        'random_sampling': args.random_sampling,
        'l1_reg': args.l1_reg,
        'l2_reg': args.l2_reg
    }
    if args.active:
        trainer = ActiveBackpropTrainer(mlp, neg_log_cost_w_l1_l2, datasets, trainer_config)
    else:
        trainer = BackpropTrainer(mlp, neg_log_cost_w_l1_l2, datasets, trainer_config)

    print "Training."
    best_validation_loss, test_score = trainer.train(args.epochs)
    print(('Optimization complete. Best validation score: %f %%. '
           'Test performace %f %%.') %
          (best_validation_loss * 100., test_score * 100.))
    end_time = time.clock()
    print 'The code ran for %.2fm' % ((end_time - start_time) / 60.)
