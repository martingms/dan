import time

import numpy as np
import theano
import theano.tensor as T

from mnist import load_data

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
    def __init__(self, input, n_in, n_nodes, W=None, b=None,
            activation=lambda x: x, dropout_rate=0.5):
        srng = T.shared_randomstreams.RandomStreams(91231) # TODO: Seed properly?
        dropout_mask = srng.binomial(n=1, p=1-dropout_rate, size=input.shape)

        # Keeps stuff on GPU
        input *= T.cast(dropout_mask, theano.config.floatX)

        super(DropoutLayer, self).__init__(input, n_in, n_nodes, W, b,
                activation)

class MLP(object):
    """TODO: Write docstring"""
    def __init__(self, rng, n_in, n_hidden_list, n_out, dropout_rate_list):
        assert len(n_hidden_list) + 1 == len(dropout_rate_list)
        ### Set up Theano variables
        self.bindex = T.lscalar()
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        ### Wire up network
        self.layers = []
        self.dropout_layers = []

        # Hidden layers
        dropout_input = self.x
        input = self.x
        for n_layer, dropout_rate in zip(n_hidden_list, dropout_rate_list):
            dropout_layer = DropoutLayer(
                input=dropout_input,
                n_in=n_in,
                n_nodes=n_layer,
                W=Layer.generate_W(rng, n_in, n_layer),
                activation=T.tanh,
                dropout_rate=dropout_rate
            )
            dropout_input = dropout_layer.output
            self.dropout_layers.append(dropout_layer)
            layer = Layer(
                input=input,
                n_in=n_in,
                n_nodes=n_layer,
                # Scaling based on dropout.
                # TODO: per layer
                W=dropout_layer.W * (1-dropout_rate),
                b=dropout_layer.b,
                activation=T.tanh
            )
            self.layers.append(layer)
            input = layer.output
            n_in = n_layer

        # Softmax output layer
        self.dropout_layers.append(DropoutLayer(
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

        #output_printed = theano.printing.Print('output')(self.layers[-1].output)
        #self.y_pred = T.argmax(output_printed, axis=1)
        self.y_pred = T.argmax(self.layers[-1].output, axis=1)

        self.params = [param for layer in self.dropout_layers
        #self.params = [param for layer in self.layers
                             for param in layer.params]

    def neg_log_likelihood(self, y):
        return -T.mean(T.log(self.dropout_layers[-1].output)[T.arange(y.shape[0]), y])
        #return -T.mean(T.log(self.layers[-1].output)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if not y.dtype.startswith('int'):
            raise NotImplementedError()
        #y_printed = theano.printing.Print('y')(y)
        #y_pred_printed = theano.printing.Print('y_pred')(self.y_pred)
        #result_printed = theano.printing.Print('result')(T.mean(T.neq(y_pred_printed, y_printed)))
        return T.mean(T.neq(self.y_pred, y))

    def L1(self):
        return sum([abs(layer.W).sum() for layer in self.layers])

    def L2(self):
        return sum([(layer.W ** 2).sum() for layer in self.layers])

    def train(self, train_set, valid_set, test_set, learning_rate=0.01,
            L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=20,
            patience=10000, patience_increase=2, improvement_threshold=0.995):
        train_set_x, train_set_y = train_set
        valid_set_x, valid_set_y = valid_set
        test_set_x, test_set_y = test_set

        ### Set up Theano functions
        cost = (
            self.neg_log_likelihood(self.y)
            + L1_reg * self.L1()
            + L2_reg * self.L2()
        )
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
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
        
        self.validate_func = theano.function(
            inputs=[self.bindex],
            outputs=self.errors(self.y),
            givens={
                self.x: valid_set_x[self.bindex * batch_size:(self.bindex + 1) * batch_size],
                self.y: valid_set_y[self.bindex * batch_size:(self.bindex + 1) * batch_size]
            }
        )

        self.test_func = theano.function(
            inputs=[self.bindex],
            outputs=self.errors(self.y),
            givens={
                self.x: test_set_x[self.bindex * batch_size:(self.bindex + 1) * batch_size],
                self.y: test_set_y[self.bindex * batch_size:(self.bindex + 1) * batch_size]
            }
        )

        #theano.printing.pydotprint(train_func, outfile="train_func.png",
        #        var_with_name_simple=True)
        #theano.printing.pydotprint(self.validate_func, outfile="validate_func.png",
        #        var_with_name_simple=True)
  
        ### 
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
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
                    validation_losses = [self.validate_func(i) for i
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
                        test_losses = [self.test_func(i) for i
                                       in xrange(n_test_batches)]
                        test_score = np.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, bindex + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))

if __name__ == '__main__':
    print "Loading dataset."
    datasets = load_data('mnist.pkl.gz')

    rng = np.random.RandomState(1234)

    print "Generating model."
    mlp = MLP(rng, 28*28, [500], 10, [0.0, 0.0])

    print "Training."
    mlp.train(datasets[0], datasets[1], datasets[2], L1_reg=0.0, L2_reg=0.0, n_epochs=10000)
    #mlp.train(datasets[0], datasets[1], datasets[2])
