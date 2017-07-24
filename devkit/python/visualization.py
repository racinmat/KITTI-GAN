# Autogenerated with SMOP
from smop.core import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def visualizationInit(image_dir=None, frame=None):
    # create figure using size of first image in repository
    img = mpimg.imread('{:s}/{:010d}.png'.format(image_dir, frame))
    # plt.figure(1, figsize=(np.dot(0.8, size(img, 2)), np.dot(0.8 * 2, size(img, 1))))
    plt.figure()
    h = {0: {}, 1: {}}
    h[0]['axes'] = plt.axes([0, 0.5, 1, 0.5])
    h[1]['axes'] = plt.axes([0, 0, 1, 0.5])

    return h


def visualizationUpdate(image_dir=None, h=None, img_idx=None, nimages=None):
    img = mpimg.imread('{:s}/{:010d}.png'.format(image_dir, img_idx))
    plt.cla()
    h[0]['axes'].imshow(img)
    # h[0]['axes'].hold('on')
    h[1]['axes'].imshow(img)
    # h[1]['axes'].hold('on')
    h[0]['axes'].text(size(img, 2) / 2, 3, s='2D Bounding Boxes', color='g', horizontalalignment='center', verticalalignment='top', fontsize=14, fontweight='bold',
                      backgroundcolor='black')
    h[1]['axes'].text(size(img, 2) / 2, 3, s='3D Bounding Boxes', color='g', horizontalalignment='center', verticalalignment='top', fontsize=14, fontweight='bold',
                      backgroundcolor='black')
    h[0]['axes'].text(0, 0, s='Not occluded', color='g', horizontalalignment='left', verticalalignment='top', fontsize=14, fontweight='bold', backgroundcolor='black')
    h[0]['axes'].text(0, 30, s='Partly occluded', color='y', horizontalalignment= 'left',
                      verticalalignment='top', fontsize=14, fontweight='bold', backgroundcolor='black')
    h[0]['axes'].text(0, 60, s='Fully occluded', color='r', horizontalalignment='left',
                      verticalalignment='top', fontsize=14, fontweight='bold', backgroundcolor='black')
    h[0]['axes'].text(0, 90, s='Unknown', color='w', horizontalalignment='left',
                      verticalalignment='top', fontsize=14, fontweight='bold', backgroundcolor='black')
    h[0]['axes'].text(size(img, 2), 0, s='frame {:d}/{:d}'.format(img_idx, nimages - 1), color='g',
                      horizontalalignment='right', verticalalignment='top', fontsize=14, fontweight='bold',
                      backgroundcolor='black')
    # h[1]['axes'].text(size(img, 2) / 2, size(img, 1),
    #                   s='\'SPACE\': Next Image  |  \'-\': Previous Image  |  \'x\': +100  |  \'y\': -100 | \'q\': quit',
    #                   color='g', horizontalalignment='center', verticalalignment='bottom',
    #                   fontsize=14, fontweight='bold', backgroundcolor='black')
