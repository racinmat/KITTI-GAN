import glob

import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
from devkit.python.utils import loadFromFile, transform_to_range
from utils import tracklet_to_bounding_box, is_tracklet_seen

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


def load_tracklets(base_dir=None):
    cam = 2

    # get image sub-directory
    # read tracklets for the selected sequence
    tracklets = readTracklets(base_dir + '/tracklet_labels.xml')
    return tracklets


def is_for_dataset(tracklet, frame):
    # only cars in dataset
    if tracklet['objectType'] != 'Car':
        return False

    pose_idx = frame - tracklet['first_frame']
    pose = tracklet['poses_dict'][pose_idx]

    # filter out occluded tracklets
    if pose['occlusion'] != 0:
        return False

    treshold = 5 * pi / 180
    # filter out cars with high rotation
    if pose['rz'] > treshold or pose['rz'] < - treshold:
        return False

    return True


if __name__ == '__main__':
    dirs = [
        './data/2011_09_26/2011_09_26_drive_0009_sync',
        './data/2011_09_26/2011_09_26_drive_0015_sync',
        './data/2011_09_26/2011_09_26_drive_0023_sync',
        './data/2011_09_26/2011_09_26_drive_0032_sync',
    ]

    tracklets = []
    # for dir in dirs:
    #     tracklets += load_tracklets(base_dir=dir)
    #
    # print('data loaded')
    #
    # rz = np.array([wrapToPi(value) for tracklet in tracklets for value in tracklet['poses'][5]])
    # rz = rz * 180 / pi
    # print('data transformed')
    #
    # # the histogram of the data
    # nbins = 360
    # # nbins = 360 * 2
    # # nbins = 360 * 3
    # # n, bins, patches = plt.hist(rz, nbins, normed=1, facecolor='green', alpha=0.75)
    # n, bins, patches = plt.hist(x=rz, bins=nbins)
    # # n, bins, patches = plt.hist(x=rz)
    #
    # arg_max = n.argmax()
    # max_frequency = n[arg_max]
    #
    # plt.xlabel('yaw angle')
    # plt.ylabel('frequency')
    # plt.title('frequency of yaw angles of bounding box')
    # plt.axis([-180, 180, 0, ceil(max_frequency / 100) * 100])
    # # plt.grid(True)
    #
    # # plt.show()
    #
    # print('data plotted')
    #
    # plt.savefig('angles.png')
    #
    # print('plot persisted')
    #
    # print("most frequent angle: " + str(round(bins[arg_max])) + " with frequency: " + str(max_frequency))
    # print(str(len(rz)) + " number of samples in total")


    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]
    drive_dir = './data/2011_09_26/2011_09_26_'
    calib_dir = './data/2011_09_26'

    cam = 2

    # names = {
    #     0: 'tx',
    #     1: 'ty',
    #     2: 'tz',
    #     3: 'rx',
    #     4: 'ry',
    #     5: 'rz',
    #     6: 'state',
    #     7: 'occlusion',
    #     8: 'occlusion_kf',
    #     9: 'truncation',
    #     10: 'amt_occlusion',
    #     11: 'amt_occlusion_kf',
    #     12: 'amt_border_l',
    #     13: 'amt_border_r',
    #     14: 'amt_border_kf'
    # }
    #
    # posesData = np.empty((15, 0), dtype=float)
    # for i, drive in enumerate(drives):
    #     dir = drive_dir + drive
    #     tracklets = load_tracklets(base_dir=dir)
    #     for j, tracklet in enumerate(tracklets):
    #         posesData = np.concatenate((posesData, tracklet['poses']), axis=1)
    #
    # nbins = 100
    # fig = plt.figure()
    # for i in range(0, 15):
    #     fig.add_subplot(15, 1, i+1)
    #     plt.hist(x=posesData[i, :], bins=nbins)
    #     plt.title('hist of ' + names[i])
    #
    # fig.set_figheight(20)
    # fig.subplots_adjust(hspace=2)
    # plt.savefig('hists.png')

    # velo = np.empty((0, 4))
    # maximum = 0
    # minimum = 0
    #
    # for i, drive in enumerate(drives):
    #     current_dir = drive_dir + drive
    #     image_dir = current_dir + '/image_{:02d}/data'.format(cam)
    #     # get number of images for this dataset
    #     frames = len(glob.glob(image_dir + '/*.png'))
    #     start = 0
    #     end = frames
    #
    #     tracklets = load_tracklets(base_dir=current_dir)
    #     for frame in range(start, end):
    #         # percentage printing
    #         percent = 20
    #         part = int(((100 * frame) / frames) / percent)
    #         previous = int(((100 * (frame - 1)) / frames) / percent)
    #         if part - previous > 0:
    #             print(str(percent * part) + '% extracted.')
    #
    #         fname = '{:s}/velodyne_points/data/{:010d}.bin'.format(current_dir, frame)
    #         if not os.path.isfile(fname):
    #             continue
    #
    #         velo = loadFromFile(fname, 4, np.float32)
    #         local_max = max(velo[:, 0])
    #         local_min = min(velo[:, 0])
    #         if local_max > maximum:
    #             maximum = local_max
    #             print(maximum)
    #         if local_min < minimum:
    #             minimum = local_min
    #             print(minimum)
    #
    # print('maximum:' + str(maximum))
    # print('minimum:' + str(minimum))

    # print(transform_to_range(5, 80, 0, 1, 5))
    # print(transform_to_range(5, 80, 0, 1, 80))
    # print(transform_to_range(5, 80, 0, 1, 50))

    # data_dir = 'data/extracted'
    # sizes_x = np.empty((1, 0))
    # sizes_y = np.empty((1, 0))
    # for filename in glob.glob(data_dir + '/tracklets_points_image_grayscale_bg_white_drive_*.data'):
    #     print("processing: " + filename)
    #     file = open(filename, 'rb')
    #     data = pickle.load(file)
    #     file.close()
    #     for pair in data:
    #         size = pair['y'].shape
    #         sizes_x = np.append(sizes_x, size[0])
    #         sizes_y = np.append(sizes_y, size[1])
    #
    # nbins = 500
    # fig = plt.figure()
    # fig.add_subplot(2, 1, 1)
    # plt.hist(x=sizes_x, bins=nbins)
    # plt.title('hist of x')
    #
    # fig.add_subplot(2, 1, 2)
    # plt.hist(x=sizes_y, bins=nbins)
    # plt.title('hist of y')
    #
    # plt.savefig('image_size_hists.png')
    #
    # print('min x: ' + str(sizes_x.min()))
    # print('min y: ' + str(sizes_y.min()))
    # print('max x: ' + str(sizes_x.max()))
    # print('max y: ' + str(sizes_y.max()))

    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]
    drive_dir = './data/2011_09_26/2011_09_26_'
    calib_dir = './data/2011_09_26'

    cam = 2

    data = np.empty((0, 1))

    for i, drive in enumerate(drives):
        current_dir = drive_dir + drive
        image_dir = current_dir + '/image_{:02d}/data'.format(cam)
        # get number of images for this dataset
        frames = len(glob.glob(image_dir + '/*.png'))
        start = 0
        end = frames
        tracklets = load_tracklets(base_dir=current_dir)
        for frame in range(start, end):
            for j, tracklet in enumerate(tracklets):
                if not is_tracklet_seen(tracklet=tracklet, frame=frame, calib_dir=calib_dir, cam=cam):
                    continue

                if not is_for_dataset(tracklet=tracklet, frame=frame):
                    continue

                corners, t, rz, box, corners_3D = tracklet_to_bounding_box(tracklet=tracklet,
                                                               cam=cam,
                                                               frame=frame,
                                                               calib_dir=calib_dir)

                corner_ldf = corners_3D[:, 7]
                corner_urb = corners_3D[:, 1]
                distance = corner_ldf[2]

                sample = corner_ldf[2]
                data = np.vstack((data, sample))

    fig = plt.figure()
    nbins = 500
    plt.title('frequency of distances of bounding boxes')
    plt.xlabel('distance of bb left corner')
    plt.ylabel('frequency')
    plt.hist(x=data, bins=nbins)
    plt.savefig('hist-distance.png', format='png')
