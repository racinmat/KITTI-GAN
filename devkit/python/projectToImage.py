from smop.core import *
import numpy as np


def projectToImage(pts_3D=None, K=None):
    # PROJECTTOIMAGE projects 3D points in given coordinate system in the image
    # plane using the given calibration matrix K.
    # project in image
    pts_2D = np.dot(K, pts_3D[0:3, :])
    # scale projected points
    pts_2D[0, :] = pts_2D[0, :] / pts_2D[2, :]
    pts_2D[1, :] = pts_2D[1, :] / pts_2D[2, :]
    return pts_2D[0:2, :]
