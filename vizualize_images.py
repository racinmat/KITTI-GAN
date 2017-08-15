import matplotlib
# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

from devkit.python.utils import loadFromFile, transform_to_range, load_image
from utils import get_pointcloud, tracklet_to_bounding_box, pointcloud_to_image, bounding_box_to_image, \
    pointcloud_to_figure, figure_to_image
import glob
from devkit.python.readTracklets import readTracklets
import numpy as np
from devkit.python.wrapToPi import wrapToPi
from math import cos, sin, pi, ceil
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import operator
import os
import pickle
from PIL import Image, ImageOps

data_dir = 'data/extracted'
sizes_x = np.empty((1, 0))
sizes_y = np.empty((1, 0))
drives = [
    'drive_0009_sync',
    'drive_0015_sync',
    'drive_0023_sync',
    'drive_0032_sync',
]

input_prefix = 'tracklets_points_normalized_'
resolution = '32_32'
# resolution = '64_64'

directory = 'images/' + resolution
directory = directory + '/normalized/'

cam = 2
drive_dir = './data/2011_09_26/2011_09_26_'
calib_dir = './data/2011_09_26'

if not os.path.exists(directory):
    os.makedirs(directory)

for i, drive in enumerate(drives):
    filename = data_dir + '/' + input_prefix + drive + '_' + resolution + '.data'
    print("processing: " + filename)
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    for j, pair in enumerate(data):
        img = Image.fromarray(pair['y'])

        # metadata loading and kitti image generating
        metadata = pair['metadata']
        current_dir = drive_dir + drive
        image_dir = current_dir + '/image_{:02d}/data'.format(cam)
        frame = metadata['frame']
        tracklet = metadata['tracklet']

        corners, t, rz, box, corners_3D, pose_idx = tracklet_to_bounding_box(tracklet=tracklet,
                                                                             cam=cam,
                                                                             frame=frame,
                                                                             calib_dir=calib_dir)

        kitti_img = load_image('{:s}/image_{:02d}/data/{:010d}.png'.format(current_dir, cam, frame))
        velo = metadata['velo']
        velo_img = metadata['velo_img']
        fig, ax = pointcloud_to_figure(velo, velo_img, kitti_img, False)
        bounding_box_to_image(ax=ax, box=box, occlusion=tracklet['poses_dict'][pose_idx]['occlusion'],
                              object_type=tracklet['objectType'])
        buf, im = figure_to_image(fig)

        # save images
        img.save(directory + drive + '_' + str(j) + '.png')
        im.save(directory + drive + '_' + str(j) + '_src.png')
        buf.close()
