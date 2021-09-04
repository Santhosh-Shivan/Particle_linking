import numpy as np


def tonehot(indices, max_value, axis=-1):
    """
    Create a one-hot vector from indices.
    """
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)

    return one_hot
