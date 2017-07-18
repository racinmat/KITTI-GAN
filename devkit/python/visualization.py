# Autogenerated with SMOP
from smop.core import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

@function
def visualization(mode=None, image_dir=None, varargin=None):
    varargin = visualization.varargin

    if 'init' == mode:
        # create figure using size of first image in repository
        img = mpimg.imread('{:s}/{:010d}.png'.format(image_dir, 0))
        fig = plt.figure(1, figsize=(dot(0.8, size(img, 2)), dot(dot(0.8, 2), size(img, 1))))
        h = {0: {}, 1: {}}
        h[0]['axes'] = plt.axes([0, 0.5, 1, 0.5])
        h[1]['axes'] = plt.axes([0, 0, 1, 0.5])

        result = {1: h}
        return result

    else:
        if 'update' == mode:
            # unpack input arguments
            h = varargin[1]
            # /opt/project/devkit/matlab/visualization.m:20
            img_idx = varargin[2]
            # /opt/project/devkit/matlab/visualization.m:21
            nimages = varargin[3]
            # /opt/project/devkit/matlab/visualization.m:22
            img = imread(sprintf('%s/%010d.png', image_dir, img_idx))
            # /opt/project/devkit/matlab/visualization.m:25
            cla(h[1].axes)
            cla(h[2].axes)
            imshow(img, 'parent', h[1].axes)
            axis(h[1].axes, 'image', 'off')
            hold(h[1].axes, 'on')
            imshow(img, 'parent', h[2].axes)
            axis(h[2].axes, 'image', 'off')
            hold(h[2].axes, 'on')
            text(size(img, 2) / 2, 3, sprintf('2D Bounding Boxes'), 'parent', h[1].axes, 'color', 'g',
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold',
                 'BackgroundColor', 'black')
            text(size(img, 2) / 2, 3, sprintf('3D Bounding Boxes'), 'parent', h[2].axes, 'color', 'g',
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold',
                 'BackgroundColor', 'black')
            text(0, 0, 'Not occluded', 'parent', h[1].axes, 'color', 'g', 'HorizontalAlignment', 'left',
                 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'black')
            text(0, 30, 'Partly occluded', 'parent', h[1].axes, 'color', 'y', 'HorizontalAlignment', 'left',
                 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'black')
            text(0, 60, 'Fully occluded', 'parent', h[1].axes, 'color', 'r', 'HorizontalAlignment', 'left',
                 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'black')
            text(0, 90, 'Unknown', 'parent', h[1].axes, 'color', 'w', 'HorizontalAlignment', 'left',
                 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'black')
            text(size(img, 2), 0, sprintf('frame %d/%d', img_idx, nimages - 1), 'parent', h[1].axes, 'color', 'g',
                 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold',
                 'BackgroundColor', 'black', 'Interpreter', 'none')
            text(size(img, 2) / 2, size(img, 1), sprintf(
                '\'SPACE\': Next Image  |  \'-\': Previous Image  |  \'x\': +100  |  \'y\': -100 | \'q\': quit'),
                 'parent', h[2].axes, 'color', 'g', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom',
                 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'black')
