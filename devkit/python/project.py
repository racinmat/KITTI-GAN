from smop.core import *
from numpy.matlib import repmat

def project(p_in=None, T=None):
    # dimension of data and projection matrix
    dim_norm = size(T, 1)
    dim_proj = size(T, 2)
    # do transformation in homogenuous coordinates
    p2_in = copy(p_in)
    if size(p2_in, 2) < dim_proj:
        p2_in[:, dim_proj] = 1

    p2_out = (np.dot(T, p2_in.T)).T
    # normalize homogeneous coordinates:
    p_out = p2_out[:, 0:dim_norm - 1] / repmat(p2_out[:, dim_norm - 1], 2, 1).T
    return p_out