from smop.core import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def drawBox2D(h=None, box=None, occlusion=None, object_type=None):
    # set styles for occlusion and truncation
    occ_col = ['w', 'g', 'y', 'r']
    # show rectangular bounding boxes
    h[0]['axes'].add_patch(Rectangle((box['x1'], box['y1']), width=(box['x2'] - box['x1'] + 1), height=(box['y2'] - box['y1'] + 1), edgecolor=occ_col[int(occlusion) + 1], linewidth=6, fill=False))
    h[0]['axes'].add_patch(Rectangle((box['x1'], box['y1']), width=(box['x2'] - box['x1'] + 1), height=(box['y2'] - box['y1'] + 1), edgecolor='k', linewidth=2, fill=False))
    # plot label
    label_text = '{:s}'.format(object_type)
    x = (box['x1'] + box['x2']) / 2
    y = box['y1']
    h[0]['axes'].text(x, np.max((y - 5, 40)), label_text, color=occ_col[int(occlusion) + 1], backgroundcolor='k', horizontalalignment='center', verticalalignment='bottom', fontweight='bold', fontsize=16)
