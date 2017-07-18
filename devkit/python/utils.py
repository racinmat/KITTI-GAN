import numpy as np
import os


def readVariable(data=None, name=None, M=None, N=None):

    if name not in data:
        return []

    if M != 1 or N != 1:
        values = np.array(data[name].split(), dtype=float)
        values = values.reshape(M, N)
        return values
    else:
        return data[name]


def loadFromFile(fname, columns, dtype):
    # result = np.empty(shape=(0, columns), dtype=dtype)
    #
    # with open(fname, 'rb') as f:
    #     fsize = os.fstat(f.fileno()).st_size
    #
    #     # while we haven't yet reached the end of the file...
    #     while f.tell() < fsize:
    #         # get the array contents
    #         row = np.fromfile(f, dtype, columns)
    #         result = np.vstack((result, row))
    #
    # return result

    with open(fname, 'rb') as f:
        result = np.fromfile(f, dtype).reshape((-1, columns))
    return result
