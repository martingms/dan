import time

import numpy as np
import theano
import theano.tensor as T

from mnist import load_data

class Layer(object):
    """A generic MLP layer"""
    def __init__(self, input, n_in, n_nodes, W=None, b=None,
            activation=lambda x: x):
        self.input = input

        if W == None:
            W = np.zeros((n_in, n_nodes), dtype=theano.config.floatX)
        if b == None:
            b = np.zeros((n_nodes,), dtype=theano.config.floatX)

        # Do we want to borrow here?
        self.W = theano.shared(value=W, name='W', borrow=True)
        self.b = theano.shared(value=b, name='b', borrow=True)

        self.output = activation(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]

class MLP(object):
    """TODO: Write docstring"""
    def __init__(self, rng, n_in, n_hidden, n_out):
        ### Set up Theano variables
        self.bindex = T.lscalar()
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        ### Wire up network
        self.hidden_layer = Layer(
            input=self.x,
            n_in=n_in,
            n_nodes=n_hidden,
            W=np.asarray(
                # Numbers from:
                # Y. Bengio, X. Glorot, Understanding the difficulty of
                # training deep feedforward neuralnetworks, AISTATS 2010
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_hidden)),
                    high=np.sqrt(6. / (n_in + n_hidden)),
                    size=(n_in, n_hidden)
                ),
                dtype=theano.config.floatX
            ),
            activation=T.tanh
        )

        self.log_reg_layer = Layer(
            input=self.hidden_layer.output,
            n_in=n_hidden,
            n_nodes=n_out,
            activation=T.nnet.softmax
        )

        self.y_pred = T.argmax(self.log_reg_layer.output, axis=1)

        ### Set up Theano functions

        #self.test_func = theano.function(
        #    inputs=[x, y],
        #    outputs=self.errors(y)
        #)

    def neg_log_likelihood(self, y):
        return -T.mean(T.log(self.log_reg_layer.output)[T.arange(y.shape[0]), y]) 

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def L1(self):
        return abs(self.hidden_layer.W).sum() + abs(self.log_reg_layer.W).sum()

    def L2(self):
        return (self.hidden_layer.W ** 2).sum() + (self.log_reg_layer.W ** 2).sum()

    def train(self, train_set, valid_set, learning_rate=0.01, L1_reg=0.00,
            L2_reg=0.0001, n_epochs=1000, batch_size=20, patience=10000,
            patience_increase=2, improvement_threshold=0.995):
        train_set_x, train_set_y = train_set
        valid_set_x, valid_set_y = valid_set

        ### Set up Theano train and validate functions
        cost = (
            self.neg_log_likelihood(self.y)
            + L1_reg * self.L1()
            + L2_reg * self.L2()
        )
        params = self.hidden_layer.params + self.log_reg_layer.params
        gparams = [T.grad(cost, param) for param in params]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(params, gparams)
        ]

        train_func = theano.function(
            inputs=[self.bindex],
            outputs=cost,
            updates=updates,
            givens={
                self.x: train_set_x[self.bindex * batch_size:(self.bindex + 1) * batch_size],
                self.y: train_set_y[self.bindex * batch_size:(self.bindex + 1) * batch_size]
            }
        )
        
        validate_func = theano.function(
            inputs=[self.bindex],
            outputs=self.errors(self.y),
            givens={
                self.x: valid_set_x[self.bindex * batch_size:(self.bindex + 1) * batch_size],
                self.y: valid_set_y[self.bindex * batch_size:(self.bindex + 1) * batch_size]
            }
        )
  
        ### 
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    
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
                minibatch_avg_cost = train_func(bindex)

                iter = (epoch - 1) * n_train_batches + bindex
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_func(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%, iter %i, patience %i' %
                        (
                            epoch,
                            bindex + 1,
                            n_train_batches,
                            this_validation_loss * 100.,
                            iter,
                            patience
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        #test_losses = [test_model(i) for i
                        #               in xrange(n_test_batches)]
                        #test_score = np.mean(test_losses)
                        #
                        #print(('     epoch %i, minibatch %i/%i, test error of '
                        #       'best model %f %%') %
                        #      (epoch, bindex + 1, n_train_batches,
                        #       test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance DUNNO YET %%') %
              #(best_validation_loss * 100., best_iter + 1, test_score * 100.))
              (best_validation_loss * 100., best_iter + 1))

if __name__ == '__main__':
    print "Loading dataset."
    datasets = load_data('mnist.pkl.gz')

    rng = np.random.RandomState(123123)

    print "Generating model."
    mlp = MLP(rng, 28*28, 500, 10)

    print "Training."
    mlp.train(datasets[0], datasets[1])
