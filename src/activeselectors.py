import numpy as np
import theano
import theano.tensor as T

class ActiveSelector(object):
    def __init__(self, trainer):
        self.trainer = trainer

class Random(ActiveSelector):
    def select(self, n):
        # TODO/FIXME: Objectception.. Should probably make this nicer somehow.
        if n is 1:
            return self.trainer.model.rng.randint(self.trainer.unlabeled_set_ptr)
        return self.trainer.model.rng.randint(self.trainer.unlabeled_set_ptr, size=(n,))

class ScoreSelector(ActiveSelector):
    """This class is not meant to be used directly. Rather, all score-based
    selectors can use this class to get the defined select function which is the
    same in most cases"""
    def __init__(self, trainer):
        super(ScoreSelector, self).__init__(trainer)

        # TODO: Remove, debug
        # For debugging, making it possible to plot error against
        # scores to see how different selectors behave.
        start = T.lscalar()
        stop = T.lscalar()
        target = T.matrix("target")

        output = self.trainer.model.output()
        #output = theano.printing.Print("output")(output)

        err = output - target
        #err = theano.printing.Print("err")(err)

        distance = T.sqrt(T.sum(T.sqr(err), axis=1))
        #distance = theano.printing.Print("distance")(distance)

        self.err_distance_func = theano.function(
            inputs=[start, stop],
            outputs=distance,
            givens={
                self.trainer.model.x: self.trainer.unlabeled_set_x[start:stop],
                target: self.trainer.unlabeled_set_y[start:stop]
            }
        )

        # TODO: Delete
        self.counter = 0

        def sample(result):
            return self.trainer.model.dropout_sample_output()

        samples, updates = theano.scan(
                fn=sample,
                outputs_info=T.zeros_like(self.trainer.model.dropout_sample_output()),
                n_steps=n_samples
        )
        #samples = theano.printing.Print("samples")(samples)

        mean_point = T.mean(samples, axis=0)
        #mean_point = theano.printing.Print("mean_point")(mean_point)

        diff_from_mean = samples - mean_point
        #diff_fron_mean = theano.printing.Print("diff_from_mean")(diff_from_mean)

        euclid_dist_from_mean = T.sqrt(T.sum(T.sqr(diff_from_mean), axis=2))
        #euclid_dist_from_mean = theano.printing.Print("euc_dist_from_mean")(euclid_dist_from_mean)

        variances = T.var(euclid_dist_from_mean, axis=0)
        #variances = theano.printing.Print("variances")(variances)

        self.train_score_func = theano.function(
            inputs=[start, stop],
            outputs=variances,
            givens={
                self.trainer.model.x: self.trainer.train_set_x[
                    start:stop
                ],
            },
            updates=updates
        )

    def err_distance(self, start, stop):
        return self.err_distance_func(start, stop)

    def train_score(self, start, stop):
        return self.train_score_func(start, stop)
    ##

    def select(self, n):
        self.counter += 1
        bsize = self.trainer.config['batch_size']
        # TODO/FIXME: Should probably reuse this buffer.
        scores = np.empty(
            (self.trainer.n_unlabeled_batches, bsize),
            dtype=theano.config.floatX
        )

        # TODO: Remove, debug
        errs = np.empty(
            (self.trainer.n_unlabeled_batches, bsize),
            dtype=theano.config.floatX
        )
        ##

        for bindex in xrange(self.trainer.n_unlabeled_batches):
            range = self.trainer._calc_unlabeled_batch_range(bindex)
            score = self.score_func(*range)

            # TODO: Remove, debug
            err = self.err_distance(*range)
            ##

            # The last batch can have an uneven size. In that case, we
            # pad with zeros, since they don't mess up our results with
            # np.argmax.
            if len(score) != bsize:
                score = np.pad(score, (0, bsize-len(score)), mode='constant')

            # TODO: Remove, debug
            if len(err) != bsize:
                err = np.pad(err, (0, bsize-len(err)), mode='mean') # correct?
            ##

            scores[bindex] = score
            # TODO: Remove, debug
            errs[bindex] = err

        train_scores = np.empty(
            (self.trainer.n_train_batches, bsize),
            dtype=theano.config.floatX
        )

        for bindex in xrange(self.trainer.n_train_batches):
            range = self.trainer._calc_train_batch_range(bindex)
            score = self.train_score(*range)

            # TODO: Remove, debug
            if len(score) != bsize:
                score = np.pad(score, (0, bsize-len(score)), mode='mean') # correct?
            ##
            train_scores[bindex] = score

        debug_scores = scores.flatten()
        debug_errs = errs.flatten()
        debug_train_scores = train_scores.flatten()
        #from utils import dumpcsv
        #dumpcsv("errvsvar.csv", zip(debug_scores, debug_errs))

        print "========="
        score_argmax = np.argmax(scores)
        print "argmax(scores):", score_argmax
        print "min(scores):", np.min(scores)
        print "max(scores):", debug_scores[score_argmax]
        print "mean(scores):", np.mean(scores)
        print "median(scores):", np.median(scores)
        print "========="
        err_argmax = np.argmax(errs)
        print "argmax(errs):", err_argmax
        print "min(errs):", np.min(errs)
        print "max(errs):", debug_errs[err_argmax]
        print "mean(errs):", np.mean(errs)
        print "median(errs):", np.median(errs)
        print "========="
        print "errs[argmax(scores)]:", debug_errs[score_argmax]
        print "scores[argmax(errs)]:", debug_scores[err_argmax]
        print "========="

        print "#!#!", self.counter, np.mean(scores), np.mean(errs), np.mean(train_scores)
        ##

        if n is 1:
            return np.argmax(scores)
        return np.argpartition(scores.flatten(), -n)[-n:]


class OutputEntropy(ScoreSelector):
    def __init__(self, trainer):
        super(OutputEntropy, self).__init__(trainer)

        output = self.trainer.model.output()
        entropy = -T.sum(output * T.log(output), axis=1)

        self.score_func = theano.function(
            inputs=[self.trainer.start, self.trainer.stop],
            outputs=entropy,
            givens={
                self.trainer.model.x: self.trainer.unlabeled_set_x[
                    self.trainer.start:self.trainer.stop
                ],
            }
        )

class SoftVoteEntropy(ScoreSelector):
    # TODO/FIXME: Redo like KL
    def __init__(self, trainer):
        super(SoftVoteEntropy, self).__init__(trainer)
        assert self.trainer.model.dropout, \
                "MC-sampling makes no sense without dropout."
        n_samples = self.trainer.config['n_samples']
        assert n_samples > 1, \
                "This active selector does not work with less than two samples"

        def accumulate(result):
            return result + self.trainer.model.dropout_sample_output()
        initial = self.trainer.model.dropout_sample_output()

        sample_sum, updates = theano.scan(
                fn=accumulate,
                outputs_info=initial,
                n_steps=n_samples-1
        )

        # Average P(y|x) over all committee samples.
        output = sample_sum[-1] / n_samples
        entropy = -T.sum(output * T.log(output), axis=1)

        self.score_func = theano.function(
            inputs=[self.trainer.start, self.trainer.stop],
            outputs=entropy,
            givens={
                self.trainer.model.x: self.trainer.unlabeled_set_x[
                    self.trainer.start:self.trainer.stop
                ],
            },
            updates=updates
        )

class KullbackLeiblerDivergence(ScoreSelector):
    def __init__(self, trainer):
        super(KullbackLeiblerDivergence, self).__init__(trainer)
        assert self.trainer.model.dropout, \
                "MC-sampling makes no sense without dropout."
        n_samples = self.trainer.config['n_samples']
        assert n_samples > 1, \
                "This active selector does not work with less than two samples"

        def sample(result):
            return self.trainer.model.dropout_sample_output()

        samples, updates = theano.scan(
                fn=sample,
                outputs_info=T.zeros_like(self.trainer.model.dropout_sample_output()),
                n_steps=n_samples
        )

        # Average P(y|x) over all committee samples.
        sample_sum = T.sum(samples, axis=0)
        c_avg = sample_sum / n_samples

        # Kullback-Leibler divergence between p_theta and p_c
        # Sum over all possible outputs
        kl = T.sum(samples * T.log(samples/c_avg), axis=2)

        # TODO: T.mean...
        # Sum over all samples in the committee
        kl_sum = T.sum(kl, axis=0) / n_samples

        self.score_func = theano.function(
            inputs=[self.trainer.start, self.trainer.stop],
            outputs=kl_sum,
            givens={
                self.trainer.model.x: self.trainer.unlabeled_set_x[
                    self.trainer.start:self.trainer.stop
                ],
            },
            updates=updates
        )


class SampleVariance(ScoreSelector):
    def __init__(self, trainer):
        super(SampleVariance, self).__init__(trainer)
        assert self.trainer.model.dropout, \
                "MC-sampling makes no sense without dropout."
        n_samples = self.trainer.config['n_samples']
        assert n_samples > 1, \
                "This active selector does not work with less than two samples"

        def sample(result):
            return self.trainer.model.dropout_sample_output()

        samples, updates = theano.scan(
                fn=sample,
                outputs_info=T.zeros_like(self.trainer.model.dropout_sample_output()),
                n_steps=n_samples
        )
        #samples = theano.printing.Print("samples")(samples)

        variances = T.var(samples, axis=0)
        #variances = theano.printing.Print("variances")(variances)

        mean_variance = T.mean(variances, axis=1)
        #mean_variance = theano.printing.Print("mean_variance")(mean_variance)

        self.score_func = theano.function(
            inputs=[self.trainer.start, self.trainer.stop],
            outputs=mean_variance,
            givens={
                self.trainer.model.x: self.trainer.unlabeled_set_x[
                    self.trainer.start:self.trainer.stop
                ],
            },
            updates=updates
        )


class PointwiseSampleVariance(ScoreSelector):
    def __init__(self, trainer):
        super(PointwiseSampleVariance, self).__init__(trainer)
        assert self.trainer.model.dropout, \
                "MC-sampling makes no sense without dropout."
        n_samples = self.trainer.config['n_samples']
        assert n_samples > 1, \
                "This active selector does not work with less than two samples"

        def sample(result):
            return self.trainer.model.dropout_sample_output()

        samples, updates = theano.scan(
                fn=sample,
                outputs_info=T.zeros_like(self.trainer.model.dropout_sample_output()),
                n_steps=n_samples
        )
        #samples = theano.printing.Print("samples")(samples)

        mean_point = T.mean(samples, axis=0)
        #mean_point = theano.printing.Print("mean_point")(mean_point)

        diff_from_mean = samples - mean_point
        #diff_fron_mean = theano.printing.Print("diff_from_mean")(diff_from_mean)

        euclid_dist_from_mean = T.sqrt(T.sum(T.sqr(diff_from_mean), axis=2))
        #euclid_dist_from_mean = theano.printing.Print("euc_dist_from_mean")(euclid_dist_from_mean)

        variances = T.var(euclid_dist_from_mean, axis=0)
        #variances = theano.printing.Print("variances")(variances)

        self.score_func = theano.function(
            inputs=[self.trainer.start, self.trainer.stop],
            outputs=variances,
            givens={
                self.trainer.model.x: self.trainer.unlabeled_set_x[
                    self.trainer.start:self.trainer.stop
                ],
            },
            updates=updates
        )
