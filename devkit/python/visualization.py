# Autogenerated with SMOP
from smop.core import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def visualizationInit(mode=None, image_dir=None):
    # create figure using size of first image in repository
    img = mpimg.imread('{:s}/{:010d}.png'.format(image_dir, 0))
    fig = plt.figure(1, figsize=(dot(0.8, size(img, 2)), dot(dot(0.8, 2), size(img, 1))))
    h = {0: {}, 1: {}}
    h[0]['axes'] = plt.axes([0, 0.5, 1, 0.5])
    h[1]['axes'] = plt.axes([0, 0, 1, 0.5])

    result = {1: h}
    return result


def visualizationUpdate(image_dir=None, h=None, img_idx=None, nimages=None):
    img = mpimg.imread('[:s}/{:010d}.png'.format(image_dir, img_idx))
    plt.cla(h[1].axes)
    plt.cla(h[2].axes)
    plt.imshow(img, 'parent', h[1].axes)
    plt.axis(h[1].axes, 'image', 'off')
    plt.hold(h[1].axes, 'on')
    plt.imshow(img, 'parent', h[2].axes)
    plt.axis(h[2].axes, 'image', 'off')
    plt.hold(h[2].axes, 'on')
    plt.text(size(img, 2) / 2, 3, '2D Bounding Boxes', 'parent', h[1].axes, 'color', 'g',
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold',
             'BackgroundColor', 'black')
    plt.text(size(img, 2) / 2, 3, '3D Bounding Boxes', 'parent', h[2].axes, 'color', 'g',
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold',
             'BackgroundColor', 'black')
    plt.text(0, 0, 'Not occluded', 'parent', h[1].axes, 'color', 'g', 'HorizontalAlignment', 'left',
             'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'black')
    plt.text(0, 30, 'Partly occluded', 'parent', h[1].axes, 'color', 'y', 'HorizontalAlignment', 'left',
             'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'black')
    plt.text(0, 60, 'Fully occluded', 'parent', h[1].axes, 'color', 'r', 'HorizontalAlignment', 'left',
             'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'black')
    plt.text(0, 90, 'Unknown', 'parent', h[1].axes, 'color', 'w', 'HorizontalAlignment', 'left',
             'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'black')
    plt.text(size(img, 2), 0, 'frame {:d}/{:d}'.format(img_idx, nimages - 1), 'parent', h[1].axes, 'color', 'g',
             'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold',
             'BackgroundColor', 'black', 'Interpreter', 'none')
    plt.text(size(img, 2) / 2, size(img, 1),
             '\'SPACE\': Next Image  |  \'-\': Previous Image  |  \'x\': +100  |  \'y\': -100 | \'q\': quit',
             'parent', h[2].axes, 'color', 'g', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom',
             'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'black')
