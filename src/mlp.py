import numpy as np
import theano
import theano.tensor as T

from rbm import RBM

class Layer(object):
    """A generic perceptron layer"""
    def __init__(self, input, n_in, n_nodes, W=None, b=None,
            activation=lambda x: x):
        self.input = input
        self.n_in = n_in
        self.n_nodes = n_nodes

        if W == None:
            W = np.zeros((n_in, n_nodes), dtype=theano.config.floatX)
            W = theano.shared(value=W, name='W', borrow=True)
        if b == None:
            b = np.zeros((n_nodes,), dtype=theano.config.floatX)
            b = theano.shared(value=b, name='b', borrow=True)

        self.W = W
        self.b = b

        self.activation = activation

        self.params = [self.W, self.b]

    def output(self):
        return self.activation(T.dot(self.input, self.W) + self.b)

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
        self.srng = T.shared_randomstreams.RandomStreams(self.rng.randint(2147483647))

        self.n_out = n_out

        # Hidden layers
        dropout_input = self.x
        #np.set_printoptions(threshold=np.inf)
        input = self.x #theano.printing.Print("self.x")(self.x)
        for n_layer, dropout_rate, activation_func in zip(n_hidden_list,
                dropout_rate_list, activation_list):
            dropout_layer = DropoutLayer(
                srng=self.srng,
                input=dropout_input,
                n_in=n_in,
                n_nodes=n_layer,
                W=Layer.generate_W(rng, n_in, n_layer),
                activation=activation_func,
                dropout_rate=dropout_rate
            )
            dropout_input = dropout_layer.output()
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
            input = layer.output()
            n_in = n_layer

        # Softmax output layer
        self.dropout_layers.append(DropoutLayer(
            srng=self.srng,
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

        self.y_pred = T.argmax(self.output(), axis=1)

        self.params = [param for layer in self.dropout_layers
                             for param in layer.params]

    def neg_log_likelihood(self, y):
        if not self.dropout:
            return -T.mean(T.log(self.layers[-1].output())[T.arange(y.shape[0]), y])
        return -T.mean(T.log(self.dropout_layers[-1].output())[T.arange(y.shape[0]), y])

    def errors(self):
        if self.y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', self.y.type, 'y_pred', self.y_pred.type)
            )
        if not self.y.dtype.startswith('int'):
            raise NotImplementedError()
        return T.mean(T.neq(self.y_pred, self.y))

    def output(self):
        return self.layers[-1].output()

    def dropout_sample_output(self):
        return self.dropout_layers[-1].output()

    def L1(self):
        return sum([abs(layer.W).sum() for layer in self.layers])

    def L2(self):
        return sum([(layer.W ** 2).sum() for layer in self.layers])


class DBN(MLP):
    def __init__(self, rng, n_in, n_hidden_list, n_out, dropout_rate_list,
            activation_list):
        super(DBN, self).__init__(rng, n_in, n_hidden_list, n_out,
                        dropout_rate_list, activation_list)

        self.rbm_layers = []
        for i in xrange(len(self.layers)-1):
            layer = self.dropout_layers[i]
            rbm_layer = RBM(numpy_rng=rng,
                            theano_rng=self.srng,
                            input=layer.input,
                            n_visible=layer.n_in,
                            n_hidden=layer.n_nodes,
                            W=layer.W,
                            hbias=layer.b)
            self.rbm_layers.append(rbm_layer)

class LinearMLP(MLP):
    # TODO: This should probably be rewritten to be less hacky, but who has the
    # time? Instead of reverting stuff from the base MLP class, move everything
    # common out to a superclass.
    def __init__(self, rng, n_in, n_hidden_list, n_out, dropout_rate_list,
            activation_list):
        super(LinearMLP, self).__init__(rng, n_in, n_hidden_list, n_out,
                        dropout_rate_list, activation_list)

        self.y = T.matrix('y')
        self.layers[-1].activation = lambda x: x
        self.dropout_layers[-1].activation = lambda x: x

        self.y_pred = self.output()

    def neg_log_likelihood(self, y):
        raise NotImplementedError('NLL not implemented for regression.')

    def errors(self):
        """RMSE"""
        if self.y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', self.y.type, 'y_pred', self.y_pred.type)
            )
        if not self.y.dtype.startswith('float'):
            raise NotImplementedError()

        #output = theano.printing.Print("errorsoutput")(self.rmse(self.y))
        #return output
        return self.rmse(self.y)

    def rmse(self, y):
        #y_pred = theano.printing.Print("y_pred")(self.y_pred)
        #y = theano.printing.Print("y")(y)
        #se = theano.printing.Print("se")(T.sqr((y_pred - y)))
        #mse = theano.printing.Print("mse")(T.mean(se))
        #rmse = theano.printing.Print("rmse")(T.sqrt(mse))
        #return rmse
        return T.sqrt(T.mean(T.sqr((self.y_pred - y))))
