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
        ### Set up Theano variables
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        ### Wire up network
        self.layers = []
        self.dropout_layers = []

        # Main rng used to seed shared rng. This is probably the easiest way to get determinism.
        srng = T.shared_randomstreams.RandomStreams(rng.randint(2147483647))

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
            W=self.dropout_layers[-1].W,
            b=self.dropout_layers[-1].b,
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

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if not y.dtype.startswith('int'):
            raise NotImplementedError()
        return T.mean(T.neq(self.y_pred, y))

    def output_entropy(self, y):
        output = self.layers[-1].output
        return -T.sum(output * T.log(output), axis=1)

    def L1(self):
        return sum([abs(layer.W).sum() for layer in self.layers])

    def L2(self):
        return sum([(layer.W ** 2).sum() for layer in self.layers])

    def train(self, train_set, valid_set, test_set, initial_learning_rate=0.25,
            learning_rate_decay=None, L1_reg=0.00, L2_reg=0.0001,
            n_epochs=1000, batch_size=20, perform_early_stopping=False,
            patience=10000, patience_increase=2, improvement_threshold=0.995,
            max_col_norm=15):

        # Split training set into labeled and unlabeled sets.
        # Initialize labeled pool with 240 examples (like Nguyen & Smulders 2004).
        train_set_x, train_set_y = train_set[0][:240], train_set[1][:240]
        # Pad with zeros so we don't have to resize when adding new examples to the pool.
        # How much to pad can be set to the max number of examples we want to add.
        # Erring on the side of padding too much for now.
        train_set_x = np.pad(train_set_x, ((0,len(train_set[0])-len(train_set_x)), (0,0)), mode='constant')
        train_set_y = np.pad(train_set_y, (0,len(train_set[1])-len(train_set_y)), mode='constant')
        train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))

        unlabeled_set_x, unlabeled_set_y = shared_dataset((train_set[0][240:],
                train_set[1][240:]))

        set_ptrs = {'train': 240, 'unlabeled': len(train_set[0][240:])-1}

        valid_set_x, valid_set_y = shared_dataset(valid_set)
        test_set_x, test_set_y = shared_dataset(test_set)

        n_unlabeled_batches = unlabeled_set_x.get_value(borrow=True).shape[0] / batch_size
        n_train_batches = int(math.ceil(set_ptrs['train'] / float(batch_size)))
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

        learning_rate = theano.shared(
            np.cast[theano.config.floatX](initial_learning_rate)
        )

        ### Set up Theano functions
        cost = (
            self.neg_log_likelihood(self.y)
            + L1_reg * self.L1()
            + L2_reg * self.L2()
        )
        gparams = [T.grad(cost, param) for param in self.params]

        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - learning_rate * gparam

        # Max-norm regularization
        if max_col_norm is not None:
            for param, stepped_param in updates.iteritems():
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(max_col_norm))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale

        def calc_range(bindex):
            return (bindex * batch_size, (bindex +1) * batch_size)

        start = T.lscalar()
        stop = T.lscalar()

        train_func = theano.function(
            inputs=[start, stop],
            outputs=cost,
            updates=updates,
            givens={
                self.x: train_set_x[start:stop],
                self.y: train_set_y[start:stop]
            }
        )

        if learning_rate_decay is not None:
            learning_rate_update = theano.function(
                inputs=[],
                updates={learning_rate: learning_rate * learning_rate_decay}
            )

        entropy_func = theano.function(
            inputs=[start, stop],
            outputs=self.output_entropy(self.x),
            givens={
                self.x: unlabeled_set_x[start:stop],
            }
        )

        # TODO/FIXME: Should probably be called in theano.function?
        # YES, see http://stackoverflow.com/questions/15917849/how-can-i-assign-update-subset-of-tensor-shared-variable-in-theano
        def copy_to_train_set(idx):
            # Warning: Part of a terrible hack to avoid expensive resizing of matrices.
            # Copy value at idx in unlabeled set to first free spot in training set.
            T.set_subtensor(train_set_x[set_ptrs['train']], unlabeled_set_x.get_value()[idx])
            T.set_subtensor(train_set_y[set_ptrs['train']], unlabeled_set_y.eval()[idx])
            set_ptrs['train'] += 1

            # Delete idx from unlabeled set by swapping in bottom and decreasing pointer.
            T.set_subtensor(unlabeled_set_x[idx], unlabeled_set_x.get_value()[set_ptrs['unlabeled']])
            T.set_subtensor(unlabeled_set_y[idx], unlabeled_set_y.eval()[set_ptrs['unlabeled']])
            set_ptrs['unlabeled'] -= 1
        
        validate_func = theano.function(
            inputs=[start, stop],
            outputs=self.errors(self.y),
            givens={
                self.x: valid_set_x[start:stop],
                self.y: valid_set_y[start:stop]
            }
        )

        test_func = theano.function(
            inputs=[start, stop],
            outputs=self.errors(self.y),
            givens={
                self.x: test_set_x[start:stop],
                self.y: test_set_y[start:stop]
            }
        )

        ### 
    
        validation_frequency = min(n_train_batches, patience / 2)
    
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False
        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            for bindex in xrange(n_train_batches):
                start, stop = calc_range(bindex)
                if stop > set_ptrs['train']:
                    stop = set_ptrs['train']
                minibatch_avg_cost = train_func(start, stop)

                iter = (epoch - 1) * n_train_batches + bindex
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_func(*calc_range(i)) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%, iter %i, patience %i, learning_rate %f' %
                        (
                            epoch,
                            bindex + 1,
                            n_train_batches,
                            this_validation_loss * 100.,
                            iter,
                            patience,
                            learning_rate.get_value(borrow=True)
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (this_validation_loss < best_validation_loss *
                            improvement_threshold):
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [test_func(*calc_range(i)) for i in
                                xrange(n_test_batches)]
                        test_score = np.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, bindex + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter and perform_early_stopping:
                    done_looping = True
                    break

            ### Active learning
            # Find example with highest entropy.
            # TODO/FIXME: Should this be used in a theano.function (w/scan)?
            # TODO/FIXME: Reuse this buffer!
            # TODO/FIXME: Verify correctness
            entropies = np.empty((n_unlabeled_batches, batch_size), dtype=theano.config.floatX)
            for i in xrange(n_unlabeled_batches):
                start, stop = calc_range(i)
                # If range extends further than the pointer, set to pointer.
                # + 1 because of how ranges work.
                if stop > set_ptrs['unlabeled'] + 1:
                    stop = set_ptrs['unlabeled'] + 1
                ent = entropy_func(start, stop)
                # The last batch can have an uneven size. In that case, we
                # pad with zeros, since they don't mess up our results with
                # np.argmax.
                if len(ent) != 20:
                    ent = np.pad(ent, (0, 20-len(ent)), mode='constant')
                entropies[i] = ent

            idx = np.argmax(entropies)

            # Copy that example to training set and delete from unlabeled set.
            copy_to_train_set(idx)
            n_unlabeled_batches = int(math.ceil(set_ptrs['unlabeled'] / float(batch_size)))
            n_train_batches = int(math.ceil(set_ptrs['train'] / float(batch_size)))

            if learning_rate_decay is not None:
                learning_rate_update()

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print 'The code ran for %.2fm' % ((end_time - start_time) / 60.)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=int, default=28*28)
    parser.add_argument('-o', '--output', type=int, default=10)
    parser.add_argument('-l', '--layers', type=int, nargs='+', default=[800, 800])
    parser.add_argument('-p', '--dropout-p', type=float, nargs='+', default=[0.0, 0.0, 0.0])
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-lrd', '--learning-rate-decay', type=float, default=None)
    parser.add_argument('-s', '--seed', type=int, default=int(time.time()))
    parser.add_argument('-e', '--epochs', type=int, default=1000)
    parser.add_argument('-l1', '--l1-reg', type=float, default=0.0)
    parser.add_argument('-l2', '--l2-reg', type=float, default=0.0)
    parser.add_argument('-m', '--max-col-norm', type=float, default=None)
    args = parser.parse_args()
    print args

    print "Loading dataset."
    datasets = load_data('mnist.pkl.gz')

    rng = np.random.RandomState(args.seed)

    print "Generating model."
    mlp = MLP(rng, args.input, args.layers, args.output, args.dropout_p, [T.tanh, T.tanh])

    print "Training."
    mlp.train(datasets[0], datasets[1], datasets[2], L1_reg=args.l1_reg,
            L2_reg=args.l2_reg, n_epochs=args.epochs,
            initial_learning_rate=args.learning_rate,
            learning_rate_decay=args.learning_rate_decay,
            max_col_norm=args.max_col_norm)
