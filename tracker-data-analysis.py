import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

from devkit.python.readTracklets import readTracklets
import numpy as np
from devkit.python.wrapToPi import wrapToPi
from math import cos, sin, pi
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


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
    for dir in dirs:
        tracklets += load_tracklets(base_dir=dir)

    print('data loaded')

    rz = [wrapToPi(tracklet['poses'][5, :]) * 180 / pi for tracklet in tracklets]

    print('data transformed')

    # the histogram of the data
    nbins = 360
    # n, bins, patches = plt.hist(rz, nbins, normed=1, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(rz, nbins)
    # n, bins, patches = plt.hist(rz)

    # plt.xlabel('Angles')
    # plt.ylabel('Frequency')
    # plt.title('Frequency of angles of bounding box')
    # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)

    # plt.show()

    print('data plotted')

    plt.savefig('angles.png')

    print('plot persisted')
