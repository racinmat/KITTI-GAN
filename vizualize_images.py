import glob

import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
from devkit.python.utils import loadFromFile, transform_to_range

matplotlib.use('Agg')

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

directory = 'images/' + resolution
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

        # resize image
        img.save(directory + '/normalized_' + drive + '_' + str(j) + '.png')
