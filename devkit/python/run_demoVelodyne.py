from scipy.misc import imread
from smop.core import *
from visualization import *
from project import *
from loadCalibrationRigid import *
from loadCalibrationCamToCam import *


# /opt/project/devkit/matlab/run_demoVelodyne.m


@function
def run_demoVelodyne(base_dir=None, calib_dir=None):
    varargin = run_demoVelodyne.varargin
    nargin = run_demoVelodyne.nargin

    # KITTI RAW DATA DEVELOPMENT KIT
    #
    # Demonstrates projection of the velodyne points into the image plane

    # Input arguments:
    # base_dir .... absolute path to sequence base directory (ends with _sync)
    # calib_dir ... absolute path to directory that contains calibration files

    # clear and close everything
    disp('======= KITTI DevKit Demo =======')
    # options (modify this to select your sequence)
    if nargin < 1:
        base_dir = './../../data/2011_09_26/2011_09_26_drive_0009_sync'

    if nargin < 2:
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
    img = imread('{:s}/image_{:02d}/data/{:010d}.png'.format(base_dir, cam, frame))
    fig = figure('Position', cat(20, 100, size(img, 2), size(img, 1)))
    axes('Position', cat(0, 0, 1, 1))
    imshow(img)
    hold('on')
    # load velodyne points
    fid = fopen('{:s}/velodyne_points/data/{:010d}.bin'.format(base_dir, frame), 'rb')
    # /opt/project/devkit/matlab/run_demoVelodyne.m:39
    velo = fread(fid, cat(4, inf), 'single').T
    # /opt/project/devkit/matlab/run_demoVelodyne.m:40
    velo = velo[1:5:end(), :]
    # /opt/project/devkit/matlab/run_demoVelodyne.m:41

    fclose(fid)
    # remove all points behind image plane (approximation
    idx = velo[:, 1] < 5
    # /opt/project/devkit/matlab/run_demoVelodyne.m:45
    velo[idx, :] = []
    # /opt/project/devkit/matlab/run_demoVelodyne.m:46
    # project to image plane (exclude luminance)
    velo_img = project(velo[:, 1:3], P_velo_to_img)
    # /opt/project/devkit/matlab/run_demoVelodyne.m:49
    # plot points
    cols = copy(jet)
    # /opt/project/devkit/matlab/run_demoVelodyne.m:52
    for i in arange(1, size(velo_img, 1)).reshape(-1):
        col_idx = round(dot(64, 5) / velo[i, 1])
        # /opt/project/devkit/matlab/run_demoVelodyne.m:54
        plot(velo_img[i, 1], velo_img[i, 2], 'o', 'LineWidth', 4, 'MarkerSize', 1, 'Color', cols[col_idx, :])


run_demoVelodyne()
