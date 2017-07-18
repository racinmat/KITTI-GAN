from math import pi

from smop.core import *


def wrapToPi(alpha=None):
    # wrap to [0..2*pi]
    alpha = alpha % (2 * pi)
    # wrap to [-pi..pi]
    idx = alpha > pi
    alpha[idx] = alpha[idx] - (2 * pi)
    return alpha
