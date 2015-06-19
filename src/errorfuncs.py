import theano.tensor as T

def rmse(model, y):
    if y.ndim != model.y_pred.ndim:
        raise TypeError(
            'y should have the same shape as y_pred',
            ('y', y.type, 'y_pred', model.y_pred.type)
        )
    if not y.dtype.startswith('float'):
        raise NotImplementedError()

    return model.rmse(y)
    
def meanerrors(model, y):
    if y.ndim != model.y_pred.ndim:
        raise TypeError(
            'y should have the same shape as y_pred',
            ('y', y.type, 'y_pred', model.y_pred.type)
        )
    if not y.dtype.startswith('int'):
        raise NotImplementedError()
    return T.mean(T.neq(model.y_pred, y))
