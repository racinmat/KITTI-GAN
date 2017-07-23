from smop.core import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def drawBox3D(h=None, occlusion=None, corners=None, face_idx=None, orientation=None):
    # set styles for occlusion and truncation
    occ_col = ['w', 'g', 'y', 'r']
    # draw projected 3D bounding boxes
    if corners is not None:
        for f in range(4):
            h[1]['axes'].add_line(mlines.Line2D(np.append(corners[0, face_idx[f, :]], corners[0, face_idx[f, 0]]) + 1, np.append(corners[1, face_idx[f, :]], corners[1, face_idx[f, 0]]) + 1, color=occ_col[int(occlusion) + 1], linewidth=6))
            h[1]['axes'].add_line(mlines.Line2D(np.append(corners[0, face_idx[f, :]], corners[0, face_idx[f, 0]]) + 1, np.append(corners[1, face_idx[f, :]], corners[1, face_idx[f, 0]]) + 1, color='k', linewidth=2))

    # draw orientation vector
    if orientation is not None:
        h[1]['axes'].add_line(mlines.Line2D(orientation[0, :] + 1, orientation[1, :] + 1, color='w', linewidth=6))
        h[1]['axes'].add_line(mlines.Line2D(orientation[0, :] + 1, orientation[1, :] + 1, color='k', linewidth=2))
