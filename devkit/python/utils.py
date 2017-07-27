import numpy as np
import os
import matplotlib.image as mpimg
from functools import lru_cache


def readVariable(data=None, name=None, M=None, N=None):
    if name not in data:
        return []

    if M != 1 or N != 1:
        values = np.array(data[name].split(), dtype=float)
        values = values.reshape(M, N)
        return values
    else:
        return data[name]


@lru_cache(maxsize=32)
def loadFromFile(fname, columns, dtype):
    with open(fname, 'rb') as f:
        result = np.fromfile(f, dtype).reshape((-1, columns))
    return result


def size(a, b=0):
    s = np.asarray(a).shape
    if s is ():
        return 1 if b else (1,)
    # a is not a scalar
    try:
        if b:
            return s[b-1]
        else:
            return s
    except IndexError:
        return 1


def isempty(a):
    try:
        return 0 in np.asarray(a).shape
    except AttributeError:
        return False


@lru_cache(maxsize=32)
def load_image(filename):
    return mpimg.imread(filename)
