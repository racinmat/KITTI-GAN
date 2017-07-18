import matplotlib
# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

from devkit.python.utils import loadFromFile
from scipy.misc import imread
from smop.core import *
from visualization import *
from project import *
from loadCalibrationRigid import *
from loadCalibrationCamToCam import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def run_demoVelodyne(base_dir=None, calib_dir=None):

    # KITTI RAW DATA DEVELOPMENT KIT
    #
    # Demonstrates projection of the velodyne points into the image plane

    # Input arguments:
    # base_dir .... absolute path to sequence base directory (ends with _sync)
    # calib_dir ... absolute path to directory that contains calibration files

    # clear and close everything
    disp('======= KITTI DevKit Demo =======')
    # options (modify this to select your sequence)
    if base_dir is None:
        base_dir = './../../data/2011_09_26/2011_09_26_drive_0009_sync'

    if calib_dir is None:
        calib_dir = './../../data/2011_09_26'

    cam = 2
    frame = 20

    # load calibration
    calib = loadCalibrationCamToCam(fullfile(calib_dir, 'calib_cam_to_cam.txt'))
    Tr_velo_to_cam = loadCalibrationRigid(fullfile(calib_dir, 'calib_velo_to_cam.txt'))
    # compute projection matrix velodyne->image plane
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[0:3, 0:3] = calib['R_rect'][0]
    P_velo_to_img = dot(dot(calib['P_rect'][cam], R_cam_to_rect), Tr_velo_to_cam)
    # load and display image
    img = mpimg.imread('{:s}/image_{:02d}/data/{:010d}.png'.format(base_dir, cam, frame))
    plt.figure()
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
    cols = matplotlib.cm.jet(np.arange(256)) # jet is colormap, represented by lookup table

    for i in range(size(velo_img, 1)):
        col_idx = int(round(256*5 / velo[i, 0])) - 1
        plt.plot(velo_img[i, 0], velo_img[i, 1], 'o', linewidth=4, markersize=1, color=cols[col_idx, 0:3])

    plt.imshow(img)
    plt.savefig('foo.png')

run_demoVelodyne()
