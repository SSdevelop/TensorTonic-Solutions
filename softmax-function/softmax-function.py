import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    # x = np.asarray(x, dtype=float)
    # y = np.atleast_2d(x)
    axis = 1 if x.ndim == 2 else 0
    # max_x = np.expand_dims(np.max(x, axis=axis), axis)
    # x = x - max_x
    # e_x = np.exp(x)
    # result = e_x / np.expand_dims(np.sum(e_x, axis=axis), axis)
    # return result
    max_x = np.expand_dims(np.max(x, axis=axis), axis)
    x = x - max_x
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
