from numpy.matlib import repmat
import numpy as np


def project(p_in=None, T=None):
    # dimension of data and projection matrix
    dim_norm = np.size(T, 0)
    dim_proj = np.size(T, 1)
    # do transformation in homogenuous coordinates
    p2_in = np.copy(p_in)
    if np.size(p2_in, 1) < dim_proj:
        p2_in = np.hstack((p2_in, np.ones(shape=(np.size(p_in, 0), 1))))

    p2_out = (np.dot(T, p2_in.T)).T
    # normalize homogeneous coordinates:
    p_out = p2_out[:, 0:dim_norm - 1] / repmat(p2_out[:, dim_norm - 1], 2, 1).T
    return p_out
