import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from devkit.python.load_calibration import load_calibration_rigid, load_calibration_cam_to_cam
from devkit.python.project import project
from devkit.python.utils import loadFromFile
from devkit.python.utils import size


def run_demoVelodyne(base_dir=None, calib_dir=None):
    # KITTI RAW DATA DEVELOPMENT KIT
    #
    # Demonstrates projection of the velodyne points into the image plane

    # Input arguments:
    # base_dir .... absolute path to sequence base directory (ends with _sync)
    # calib_dir ... absolute path to directory that contains calibration files

    print('======= KITTI DevKit Demo =======')
    # options (modify this to select your sequence)
    if base_dir is None:
        base_dir = './../../data/2011_09_26/2011_09_26_drive_0009_sync'

    if calib_dir is None:
        calib_dir = './../../data/2011_09_26'

    cam = 2
    frame = 20

    image_resolution = np.array([1242, 375])
    # load calibration
    calib = load_calibration_cam_to_cam(calib_dir + '/calib_cam_to_cam.txt')
    Tr_velo_to_cam = load_calibration_rigid(calib_dir + '/calib_velo_to_cam.txt')
    # compute projection matrix velodyne->image plane
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[0:3, 0:3] = calib['R_rect'][0]
    P_velo_to_img = np.dot(np.dot(calib['P_rect'][cam], R_cam_to_rect), Tr_velo_to_cam)
    # load and display image
    img = mpimg.imread('{:s}/image_{:02d}/data/{:010d}.png'.format(base_dir, cam, frame))
    fig = plt.figure()
    plt.axes([0, 0, 1, 1])

    # load velodyne points
    fname = '{:s}/velodyne_points/data/{:010d}.bin'.format(base_dir, frame)
    velo = loadFromFile(fname, 4, np.float32)
    # keep only every 5-th point for visualization
    velo = velo[0::5, :]

    # remove all points behind image plane (approximation)
    idx = velo[:, 0] < 5
    velo = velo[np.invert(idx), :]

    # project to image plane (exclude luminance)
    velo_img = project(velo[:, 0:3], P_velo_to_img)
    # plot points
    cols = matplotlib.cm.jet(np.arange(256))  # jet is colormap, represented by lookup table

    col_indices = np.round(256 * 5 / velo[:, 0]).astype(int) - 1
    plt.scatter(x=velo_img[:, 0], y=velo_img[:, 1], c=cols[col_indices, 0:3], marker='o', s=1)

    plt.savefig('velo-only-pointcloud.png')

    dpi = fig.dpi
    fig.set_size_inches(image_resolution / dpi)
    ax = plt.gca()
    ax.set_xlim((-0.5, image_resolution[0] - 0.5))
    ax.set_ylim((image_resolution[1] - 0.5, -0.5))

    plt.savefig('velo-set.png')

    white_img = mpimg.imread('../../white.png')
    plt.imshow(white_img)
    plt.savefig('velo-white.png')

    plt.imshow(img)
    plt.savefig('velo-with-image.png')


if __name__ == '__main__':
    run_demoVelodyne()
