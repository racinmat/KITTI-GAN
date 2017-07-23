from math import pi
import numpy as np


def wrapToPi(alpha):
    # wrap to [0..2*pi]
    alpha = alpha % (2 * pi)
    if type(alpha) is np.float64:
        if alpha > pi:
            alpha = alpha - (2 * pi)
    else:
        idx = alpha > pi
        alpha[idx] = alpha[idx] - (2 * pi)
    # wrap to [-pi..pi]
    # for single value, not array
    return alpha
