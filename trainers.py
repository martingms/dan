import math
import numpy as np
import theano
import theano.tensor as T

from collections import OrderedDict

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

            if new_best_validation_loss:
                test_score = self._test()
                if test_score < best_test_score:
                    print "    epoch %i, test score %f %%" % (cur_epoch, test_score * 100.)

        return best_validation_loss, best_test_score

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
        best_test_score = np.inf
        total_epoch = 0
        for i in xrange(epochs/epochs_between_copies):
            validation_loss, test_score = super(ActiveBackpropTrainer,
                    self).train(epochs_between_copies, total_epoch,
                                    best_validation_loss, best_test_score)
            total_epoch += epochs_between_copies
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
            if test_score < best_test_score:
                best_test_score = test_score

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
                idx = self.model.rng.randint(self.unlabeled_set_ptr)

            # Copy that example to training set and delete from unlabeled set.
            self._copy_to_train_set(idx)
            self.n_unlabeled_batches = \
                    int(math.ceil(
                        self.unlabeled_set_ptr \
                                / float(self.config['batch_size'])))
            self.n_train_batches = \
                    int(math.ceil(
                        self.train_set_ptr / float(self.config['batch_size'])))

        return best_validation_loss, best_test_score

