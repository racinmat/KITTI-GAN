import ruamel.yaml as yaml
from functools import lru_cache

from devkit.python.utils import readVariable, isempty
import numpy as np


@lru_cache(maxsize=32)
def loadCalibrationCamToCam(filename):
    # open file
    with open(filename, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return {}

    calib = {}

    # read corner distance
    calib['cornerdist'] = readVariable(data, 'corner_dist', 1, 1)
    # /opt/project/devkit/matlab/loadCalibrationCamToCam.m:12
    # read all cameras (maximum: 100)

    calib['S'] = np.zeros((100, 1, 2))
    calib['K'] = np.zeros((100, 3, 3))
    calib['D'] = np.zeros((100, 1, 5))
    calib['R'] = np.zeros((100, 3, 3))
    calib['T'] = np.zeros((100, 3, 1))
    calib['S_rect'] = np.zeros((100, 1, 2))
    calib['R_rect'] = np.zeros((100, 3, 3))
    calib['P_rect'] = np.zeros((100, 3, 4))

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
