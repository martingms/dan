import csv
import theano
import numpy as np
import theano.tensor as T


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, shared_y, T.cast(shared_y, 'int32')


def dumpcsv(path, data, delimiter=' ', quotechar='|', mode='wb'):
    """Dump data into a csv file. `data` must be iterable and contain the
    rows."""
    with open(path, mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter,
                quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)
        for row in data:
            writer.writerow(row)

def normalize(A, range_=(0,1)):
    """Normalize a tensor to `range_`"""
    min_ = np.min(A)
    zero_one = (A - min_)/(np.max(A) - min_)

    if range_ == (0,1):
        return zero_one

    x, y = range_
    diff = y - x
    return (zero_one * diff) + x
