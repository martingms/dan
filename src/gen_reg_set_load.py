import gzip
import cPickle

def load_data(dataset):
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_valid_test_sets = cPickle.load(f)
    f.close()

    return train_valid_test_sets
