import numpy as np
from devkit.python.loadCalibrationCamToCam import loadCalibrationCamToCam
from devkit.python.loadCalibrationRigid import loadCalibrationRigid


def loadCalibration(dir=None):
    # LOADCALIBRATION provides all needed coordinate system transformations
    # returns the pre-computed velodyne to cam (gray and color) projection

    # get the velodyne to camera calibration
    Tr_velo_to_cam = loadCalibrationRigid(dir + '/calib_velo_to_cam.txt')
    # get the camera intrinsic and extrinsic calibration
    calib = loadCalibrationCamToCam(dir + '/calib_cam_to_cam.txt')
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

    veloToCam = {}
    veloToCam[0] = np.dot(np.dot(T0, R_rect00), Tr_velo_to_cam)
    veloToCam[1] = np.dot(np.dot(T1, R_rect00), Tr_velo_to_cam)
    veloToCam[2] = np.dot(np.dot(T2, R_rect00), Tr_velo_to_cam)
    veloToCam[3] = np.dot(np.dot(T3, R_rect00), Tr_velo_to_cam)
    # calibration matrix after rectification (equal for all cameras)
    K = calib['P_rect'][1][:, 0:3]
    return veloToCam, K
