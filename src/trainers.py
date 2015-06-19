import math
import numpy as np
import theano
import theano.tensor as T

from collections import OrderedDict, Iterable

from utils import shared_dataset

class BackpropTrainer(object):
    def __init__(self, model, cost, datasets, config):
        self.model = model
        self.config = config
        self.cost = cost
        self._init_datasets(datasets)
        self._init_theano_functions()

    def _init_datasets(self, datasets):
        train_set, valid_set, test_set = datasets
        dtype = self.model.y.dtype
        if dtype.startswith('int'):
            self.train_set_x, _, self.train_set_y = shared_dataset(train_set)
            self.valid_set_x, _, self.valid_set_y = shared_dataset(valid_set)
            self.test_set_x, _, self.test_set_y = shared_dataset(test_set)
        elif dtype.startswith('float'):
            self.train_set_x, self.train_set_y, _ = shared_dataset(train_set)
            self.valid_set_x, self.valid_set_y, _ = shared_dataset(valid_set)
            self.test_set_x, self.test_set_y, _ = shared_dataset(test_set)
        else:
            raise TypeError("output datatype " + dtype + " not supported")

        batch_size = self.config['batch_size']
        self.n_train_batches = \
                self.train_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = \
                self.valid_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_test_batches = \
                self.test_set_x.get_value(borrow=True).shape[0] / batch_size

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
        validation_losses = \
                [self.validate_func(*self._calc_test_and_valid_batch_range(i))
                 for i in xrange(self.n_valid_batches)]
        return np.mean(validation_losses)

    def _test(self):
         test_losses = \
                 [self.test_func(*self._calc_test_and_valid_batch_range(i))
                  for i in xrange(self.n_test_batches)]
         return np.mean(test_losses)

    def train(self, n_epochs=1000, cur_epoch=0, best_validation_loss=np.inf,
                    best_test_score=np.inf):
        done_looping = False
        epoch = 0
        while (epoch < n_epochs and (not done_looping)):
            epoch += 1
            cur_epoch += 1
            avg_costs = self._epoch()
            validation_loss = self._validate()

            new_best_validation_loss = False
            marker = ""
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                new_best_validation_loss = True
                marker = "*"

            print(
                'epoch %i, avg training cost %f, validation error %f, learning_rate %f %s' %
                (
                    cur_epoch,
                    avg_costs,
                    validation_loss,
                    self.learning_rate.get_value(borrow=True),
                    marker
                )
            )

            if self.config['learning_rate_decay'] is not None:
                self.learning_rate_update_func()

            if new_best_validation_loss:
                test_score = self._test()
                if test_score < best_test_score:
                    best_test_score = test_score
                    print "    epoch %i, test score %f" % (cur_epoch, test_score)

        return best_validation_loss, best_test_score

class ActiveBackpropTrainer(BackpropTrainer):
    def _init_datasets(self, datasets):
        train_set, valid_set, test_set = datasets
        # Split training set into labeled and unlabeled sets.
        n_boot = self.config['n_boostrap_examples']
        train_set_x, train_set_y = train_set[0][:n_boot], train_set[1][:n_boot]
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
                shared_dataset((train_set[0][n_boot:], train_set[1][n_boot:]))

        self.train_set_ptr = n_boot
        self.unlabeled_set_ptr = len(train_set[0][n_boot:]) - 1

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

        # Copy from unlabeled to training set function
        # Warning: Part of a terrible hack to avoid expensive resizing of matrices.
        # TODO/FIXME: Make this a OrderedDict
        idx = T.lscalar()
        train_set_ptr = T.lscalar()
        unlabeled_set_ptr = T.lscalar()
        copy_updates = [
            # Copy value at idx in unlabeled set to first free spot in training set.
            (self.train_set_x,
                T.set_subtensor(
                    self.train_set_x[train_set_ptr],
                    self.unlabeled_set_x[idx])),
            (self.train_set_y_float,
                T.set_subtensor(
                    self.train_set_y_float[train_set_ptr],
                    self.unlabeled_set_y_float[idx])),
            # Delete idx from unlabeled set by swapping in bottom and decreasing pointer.
            (self.unlabeled_set_x,
                T.set_subtensor(
                    self.unlabeled_set_x[idx],
                    self.unlabeled_set_x[unlabeled_set_ptr])),
            (self.unlabeled_set_y_float,
                T.set_subtensor(
                    self.unlabeled_set_y_float[idx],
                    self.unlabeled_set_y_float[unlabeled_set_ptr]))
        ]
        # Warning: This is wrapped in `_copy_to_train_set(idx)`.
        self.copy_to_train_set_func = theano.function(
            inputs=[idx, train_set_ptr, unlabeled_set_ptr],
            updates=copy_updates
        )

    def _copy_to_train_set(self, idx):
        """ Wrapper around `copy_to_train_set_func` to also set set-pointers
        correctly. As these are not SharedVariables, they can't be updated
        within a theano.function."""
        if isinstance(idx, Iterable):
            # TODO/FIXME: Do all copies in one call to function.
            for i in idx:
                self.copy_to_train_set_func(i, self.train_set_ptr,
                                self.unlabeled_set_ptr)
                self.train_set_ptr += 1
                self.unlabeled_set_ptr -= 1
            return

        # Else..
        self.copy_to_train_set_func(idx, self.train_set_ptr,
                        self.unlabeled_set_ptr)
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

    def train(self, epochs):
        active_selector = self.config['active_selector'](self)

        best_validation_loss = np.inf
        best_test_score = np.inf
        total_epoch = 0
        for i in xrange(epochs/self.config['epochs_between_copies']):
            validation_loss, test_score = super(ActiveBackpropTrainer,
                    self).train(self.config['epochs_between_copies'],
                                total_epoch, best_validation_loss,
                                best_test_score)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
            if test_score < best_test_score:
                best_test_score = test_score

            total_epoch += self.config['epochs_between_copies']

            # Active selection
            idx = active_selector.select(self.config['n_select'])

            # Copy that example to training set and delete from unlabeled set.
            self._copy_to_train_set(idx)
            print "    copied index", idx, "from unlabeled to training set"
            self.n_unlabeled_batches = \
                    int(math.ceil(
                        self.unlabeled_set_ptr \
                                / float(self.config['batch_size'])))
            self.n_train_batches = \
                    int(math.ceil(
                        self.train_set_ptr / float(self.config['batch_size'])))

        return best_validation_loss, best_test_score

class DBNTrainer(BackpropTrainer):
    def __init__(self, model, cost, datasets, config):
        super(DBNTrainer, self).__init__(model, cost, datasets, config)
        self._init_pretraining_functions()

    def _init_pretraining_functions(self):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        #n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * self.config['batch_size']
        # ending of a batch given `index`
        batch_end = batch_begin + self.config['batch_size']

        self.pretrain_fns = []
        for rbm in self.model.rbm_layers:
            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None,
                                                 k=self.config['k'])

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.model.x: self.train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            self.pretrain_fns.append(fn)

    def pre_train(self, epochs):
        ## Pre-train layer-wise
        for i in xrange(len(self.pretrain_fns)):
            # go through pretraining epochs
            for epoch in xrange(epochs):
                # go through the training set
                c = []
                for batch_index in xrange(self.n_train_batches):
                    c.append(self.pretrain_fns[i](index=batch_index,
                                                lr=self.config['pretrain_lr']))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print np.mean(c)
