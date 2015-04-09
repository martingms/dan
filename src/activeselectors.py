import numpy as np
import theano
import theano.tensor as T

class ActiveSelector(object):
    def __init__(self, trainer):
        self.trainer = trainer

class RandomActiveSelector(ActiveSelector):
    def select(self):
        # TODO/FIXME: Objectception.. Should probably make this nicer somehow.
        return self.trainer.model.rng.randint(self.trainer.unlabeled_set_ptr)

class OutputEntropyActiveSelector(ActiveSelector):
    def __init__(self, trainer):
        super(OutputEntropyActiveSelector, self).__init__(trainer)

        output = self.trainer.model.output()
        entropy = -T.sum(output * T.log(output), axis=1)

        self.entropy_func = theano.function(
            inputs=[self.trainer.start, self.trainer.stop],
            outputs=entropy,
            givens={
                self.trainer.model.x: self.trainer.unlabeled_set_x[
                    self.trainer.start:self.trainer.stop
                ],
            }
        )

    def select(self):
        # TODO/FIXME: Should probably reuse this buffer.
        entropies = np.empty(
            (self.trainer.n_unlabeled_batches,
                self.trainer.config['batch_size']),
            dtype=theano.config.floatX
        )

        for bindex in xrange(self.trainer.n_unlabeled_batches):
            ent = self.entropy_func(
                    *self.trainer._calc_unlabeled_batch_range(bindex))
            # The last batch can have an uneven size. In that case, we
            # pad with zeros, since they don't mess up our results with
            # np.argmax.
            if len(ent) != 20:
                ent = np.pad(ent, (0, 20-len(ent)), mode='constant')
            entropies[bindex] = ent

        return np.argmax(entropies)
