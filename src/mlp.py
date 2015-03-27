import numpy as np
import theano
import theano.tensor as T


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