from math import pi
import numpy as np


def wrapToPi(alpha):
    # wrap to [0..2*pi]
    alpha = alpha % (2 * pi)
    if type(alpha) in np.ScalarType:
        if alpha > pi:
            alpha = alpha - (2 * pi)
        return alpha
    else:
        idx = alpha > pi
        alpha[idx] = alpha[idx] - (2 * pi)
        return alpha
    # wrap to [-pi..pi]
    # for single value, not array
