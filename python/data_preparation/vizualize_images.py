import matplotlib
# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

from devkit.python.utils import load_image
from python.data_utils import tracklet_to_bounding_box, is_tracklet_seen, Cache, get_pointcloud, pointcloud_to_image, \
    figure_to_image, sample_to_image
import numpy as np
import os
import pickle
from PIL import Image


if __name__ == '__main__':
    show_data_image = True
    show_metadata_image = False

    data_dir = 'data/extracted'
    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]

    # input_prefix = 'tracklets_points_normalized_'
    input_prefix = 'tracklets_photos_normalized_'
    input_suffix = ''
    # input_suffix = '_0_20'
    resolution = '32_32'
    # resolution = '64_64'

    directory = 'images/' + resolution
    # directory = directory + '/normalized/'
    directory = directory + '/normalized_photos/'

    cam = 2
    drive_dir = './data/2011_09_26/2011_09_26_'
    calib_dir = './data/2011_09_26'

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, drive in enumerate(drives):
        filename = data_dir + '/' + input_prefix + drive + '_' + resolution + input_suffix + '.data'
        file = open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        current_dir = drive_dir + drive
        samples = len(data)
        print("processing: {:s} with {:d} samples".format(filename, samples))
        for j, sample in enumerate(data):
            # percentage printing
            percent = 5
            part = int(((100 * j) / samples) / percent)
            previous = int(((100 * (j - 1)) / samples) / percent)
            if part - previous > 0:
                print(str(percent * part) + '% visualized.')

            if show_data_image:
                img = Image.fromarray(sample['y'])
            else:
                img = None

            if show_metadata_image:
                # metadata loading and kitti image generating
                metadata = sample['metadata']
                frame = metadata['frame']
                buf, im = sample_to_image(sample, cam, calib_dir, current_dir)
            else:
                im = None
                frame = None
                buf = None

            # save images
            if show_data_image:
                img.save(directory + drive + '_{:d}.png'.format(j))
            if show_metadata_image:
                im.save(directory + drive + '_{:d}_src_frame_{:d}.png'.format(j, frame))
                buf.close()
