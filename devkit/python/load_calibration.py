import numpy as np
from functools import lru_cache
from devkit.python.utils import readVariable, isempty
import os
import pickle
import sys


@lru_cache(maxsize=32)
def load_calibration_rigid(filename=None):
    # open file
    with open(filename, 'r') as stream:
        import ruamel.yaml as yaml
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return {}

    # read calibration
    R = readVariable(data, 'R', 3, 3)
    T = readVariable(data, 'T', 3, 1)
    Tr = np.concatenate((np.concatenate((R, T), axis=1), np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    return Tr


@lru_cache(maxsize=32)
def load_calibration_cam_to_cam(filename):
    # open file
    with open(filename, 'r') as stream:
        import ruamel.yaml as yaml
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return {}

    calib = {'cornerdist': readVariable(data, 'corner_dist', 1, 1), 'S': np.zeros((100, 1, 2)),
             'K': np.zeros((100, 3, 3)),
             'D': np.zeros((100, 1, 5)),
             'R': np.zeros((100, 3, 3)),
             'T': np.zeros((100, 3, 1)),
             'S_rect': np.zeros((100, 1, 2)),
             'R_rect': np.zeros((100, 3, 3)),
             'P_rect': np.zeros((100, 3, 4))
             }

    # read corner distance
    # /opt/project/devkit/matlab/load_calibration_cam_to_cam.m:12
    # read all cameras (maximum: 100)

    for cam in np.array(range(100)).reshape(-1):
        # read variables
        S_ = readVariable(data, 'S_{:02d}'.format(cam), 1, 2)
        K_ = readVariable(data, 'K_{:02d}'.format(cam), 3, 3)
        D_ = readVariable(data, 'D_{:02d}'.format(cam), 1, 5)
        R_ = readVariable(data, 'R_{:02d}'.format(cam), 3, 3)
        T_ = readVariable(data, 'T_{:02d}'.format(cam), 3, 1)
        S_rect_ = readVariable(data, 'S_rect_{:02d}'.format(cam), 1, 2)
        R_rect_ = readVariable(data, 'R_rect_{:02d}'.format(cam), 3, 3)
        P_rect_ = readVariable(data, 'P_rect_{:02d}'.format(cam), 3, 4)
        if isempty(S_) or isempty(K_) or isempty(D_) or isempty(R_) or isempty(T_):
            break
        # write calibration
        calib['S'][cam] = S_
        calib['K'][cam] = K_
        calib['D'][cam] = D_
        calib['R'][cam] = R_
        calib['T'][cam] = T_

        if (not isempty(S_rect_)) and (not isempty(R_rect_)) and (not isempty(P_rect_)):
            calib['S_rect'][cam] = S_rect_
            calib['R_rect'][cam] = R_rect_
            calib['P_rect'][cam] = P_rect_

    return calib


@lru_cache(maxsize=32)
def load_calibration(dir=None):
    version = '.'.join([str(i) for i in sys.version_info[0:3]])
    if os.path.isfile(dir + 'calib_' + '.' + version + '.cache'):
        file = open(dir + 'calib_' + '.' + version + '.cache', 'rb')
        try:
            veloToCam, K = pickle.load(file)
            file.close()
            return veloToCam, K
        except UnicodeDecodeError:
            pass

    # LOADCALIBRATION provides all needed coordinate system transformations
    # returns the pre-computed velodyne to cam (gray and color) projection

    # get the velodyne to camera calibration
    Tr_velo_to_cam = load_calibration_rigid(dir + '/calib_velo_to_cam.txt')
    # get the camera intrinsic and extrinsic calibration
    calib = load_calibration_cam_to_cam(dir + '/calib_cam_to_cam.txt')
    # create 4x4 matrix from rectifying rotation matrix
    R_rect00 = np.zeros((4, 4))
    R_rect00[0:3, 0:3] = calib['R_rect'][0]
    R_rect00[3, 3] = 1
    # compute extrinsics from first to i'th rectified camera
    T0 = np.eye(4)
    T0[0, 3] = calib['P_rect'][0][0, 3] / calib['P_rect'][0][0, 0]
    T1 = np.eye(4)
    T1[0, 3] = calib['P_rect'][1][0, 3] / calib['P_rect'][1][0, 0]
    T2 = np.eye(4)
    T2[0, 3] = calib['P_rect'][2][0, 3] / calib['P_rect'][2][0, 0]
    T3 = np.eye(4)
    T3[0, 3] = calib['P_rect'][3][0, 3] / calib['P_rect'][3][0, 0]
    # transformation: velodyne -> rectified camera coordinates

    veloToCam = {
        0: np.dot(np.dot(T0, R_rect00), Tr_velo_to_cam),
        1: np.dot(np.dot(T1, R_rect00), Tr_velo_to_cam),
        2: np.dot(np.dot(T2, R_rect00), Tr_velo_to_cam),
        3: np.dot(np.dot(T3, R_rect00), Tr_velo_to_cam)
    }
    # calibration matrix after rectification (equal for all cameras)
    K = calib['P_rect'][1][:, 0:3]

    file = open(dir + 'calib_' + '.' + version + '.cache', 'wb')
    pickle.dump((veloToCam, K), file)
    file.close()

    return veloToCam, K
