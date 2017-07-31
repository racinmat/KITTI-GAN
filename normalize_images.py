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

input_prefix = 'tracklets_points_grayscale_bg_white_'
output_prefix = 'tracklets_points_normalized_'

new_size = (32, 32)

for drive in drives:
    filename = data_dir + '/' + input_prefix + drive + '.data'
    print("processing: " + filename)
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    for i, pair in enumerate(data):
        img = Image.fromarray(pair['y'])
        ratio = min(new_size[0] / img.size[0], new_size[1] / img.size[1])

        # resize image
        # img.save('temp_orig.png')
        new_img = img.resize((round(img.size[0] * ratio), round(img.size[1] * ratio)))
        # new_img.save('temp_resized.png')

        # fill missing places with white
        white = {'L': 255, 'RGB': (255, 255, 255)}
        bg = Image.new(mode=img.mode, size=new_size, color=white[img.mode])
        bg.paste(new_img, (0, 0, new_img.size[0], new_img.size[1]))  # Not centered, top-left corner
        # bg.save('temp_resized_padded.png')

        # oif riginal image is greyscale
        # if len(pair['y'].shape) == 2:
        #     bg = bg.convert('L')
        pair['y'] = np.array(bg)
        pair['x'].append(img.size[0])
        pair['x'].append(img.size[1])

    file = open(data_dir + '/' + output_prefix + drive + '_' + str(new_size[0]) + '_' + str(new_size[1]) + '.data', 'wb')
    pickle.dump(data, file)
    file.close()
