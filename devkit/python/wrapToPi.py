from math import pi
import numpy as np


def wrapToPi(alpha):
    # wrap to [0..2*pi]
    alpha = alpha % (2 * pi)
    # wrap to [-pi..pi]
    idx = alpha > pi
    # for single value, not array
    if type(alpha) is np.float64:
        alpha = alpha - (2 * pi)
    else:
        alpha[idx] = alpha[idx] - (2 * pi)
    return alpha
