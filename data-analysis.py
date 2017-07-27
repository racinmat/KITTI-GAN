import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

from devkit.python.readTracklets import readTracklets
import numpy as np
from devkit.python.wrapToPi import wrapToPi
from math import cos, sin, pi, ceil
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import operator

def load_tracklets(base_dir=None):
    cam = 2

    # get image sub-directory
    # read tracklets for the selected sequence
    tracklets = readTracklets(base_dir + '/tracklet_labels.xml')
    return tracklets


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

    names = {
        0: 'tx',
        1: 'ty',
        2: 'tz',
        3: 'rx',
        4: 'ry',
        5: 'rz',
        6: 'state',
        7: 'occlusion',
        8: 'occlusion_kf',
        9: 'truncation',
        10: 'amt_occlusion',
        11: 'amt_occlusion_kf',
        12: 'amt_border_l',
        13: 'amt_border_r',
        14: 'amt_border_kf'
    }

    posesData = np.empty((15, 0), dtype=float)
    for i, drive in enumerate(drives):
        dir = drive_dir + drive
        tracklets = load_tracklets(base_dir=dir)
        for j, tracklet in enumerate(tracklets):
            posesData = np.concatenate((posesData, tracklet['poses']), axis=1)

    nbins = 100
    fig = plt.figure()
    for i in range(0, 15):
        fig.add_subplot(15, 1, i+1)
        plt.hist(x=posesData[i, :], bins=nbins)
        plt.title('hist of ' + names[i])

    fig.set_figheight(20)
    fig.subplots_adjust(hspace=2)
    plt.savefig('hists.png')
