from devkit.python.loadCalibration import loadCalibration
from functools import lru_cache
from devkit.python.loadCalibrationCamToCam import loadCalibrationCamToCam
from devkit.python.loadCalibrationRigid import loadCalibrationRigid
from devkit.python.wrapToPi import wrapToPi
import numpy as np
from devkit.python.projectToImage import projectToImage
from math import cos, sin


@lru_cache(maxsize=64)
def get_corners(w, h, l):
    corners = {
        'x': [l / 2,   l / 2, - l / 2, - l / 2, l / 2,   l / 2, - l / 2, - l / 2],
        'y': [w / 2, - w / 2, - w / 2,   w / 2, w / 2, - w / 2, - w / 2,   w / 2],
        'z': [0,           0,       0,       0,     h,       h,       h,       h]
    }
    return corners


def get_corners_and_orientation(corners, rz, l, t, veloToCam, cam):
    R = rz_to_R(rz)
    corners_3D = np.dot(R, [corners['x'], corners['y'], corners['z']])
    corners_3D[0, :] = corners_3D[0, :] + t[0]
    corners_3D[1, :] = corners_3D[1, :] + t[1]
    corners_3D[2, :] = corners_3D[2, :] + t[2]
    corners_3D = np.dot(veloToCam[cam], np.vstack((corners_3D, np.ones((1, np.size(corners_3D, 1))))))
    orientation_3D = np.dot(R, [[0.0, 0.7 * l], [0.0, 0.0], [0.0, 0.0]])
    orientation_3D[0, :] = orientation_3D[0, :] + t[0]
    orientation_3D[1, :] = orientation_3D[1, :] + t[1]
    orientation_3D[2, :] = orientation_3D[2, :] + t[2]
    orientation_3D = np.dot(veloToCam[cam], np.vstack((orientation_3D, np.ones((1, np.size(orientation_3D, 1))))))
    return corners_3D, orientation_3D


@lru_cache(maxsize=32)
def get_P_velo_to_img(calib_dir, cam):
    # load calibration
    calib = loadCalibrationCamToCam(calib_dir + '/calib_cam_to_cam.txt')
    Tr_velo_to_cam = loadCalibrationRigid(calib_dir + '/calib_velo_to_cam.txt')
    # compute projection matrix velodyne->image plane
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[0:3, 0:3] = calib['R_rect'][0]
    P_velo_to_img = np.dot(np.dot(calib['P_rect'][cam], R_cam_to_rect), Tr_velo_to_cam)
    return P_velo_to_img


# @cache_bb.memoize
def tracklet_to_bounding_box(tracklet, cam, frame, calib_dir):
    veloToCam, K = loadCalibration(dir=calib_dir)
    corners = get_corners(w=tracklet['w'], h=tracklet['h'], l=tracklet['l'])

    pose_idx = frame - tracklet['first_frame']
    l = tracklet['l']
    t = [tracklet['poses_dict'][pose_idx]['tx'], tracklet['poses_dict'][pose_idx]['ty'],
         tracklet['poses_dict'][pose_idx]['tz']]
    rz = wrapToPi(tracklet['poses_dict'][pose_idx]['rz'])
    corners_3D, orientation_3D = get_corners_and_orientation(corners=corners, rz=rz, l=l, t=t,
                                                             veloToCam=veloToCam, cam=cam)
    corners_2D = projectToImage(corners_3D, K)
    box = {'x1': min(corners_2D[0, :]),
           'x2': max(corners_2D[0, :]),
           'y1': min(corners_2D[1, :]),
           'y2': max(corners_2D[1, :])}

    return corners, t, rz, box, corners_3D


def rz_to_R(rz):
    R = [[cos(rz), - sin(rz), 0],
         [sin(rz), cos(rz), 0],
         [0, 0, 1]]
    return R


def is_tracklet_seen(tracklet, frame, calib_dir, cam):
    veloToCam, K = loadCalibration(calib_dir)
    image_resolution = np.array([1242, 375])

    pose_idx = frame - tracklet['first_frame']
    # only draw tracklets that are visible in current frame
    if pose_idx < 0 or pose_idx > (np.size(tracklet['poses'], 1) - 1):
        return False

    corners = get_corners(w=tracklet['w'], h=tracklet['h'], l=tracklet['l'])
    l = tracklet['l']
    pose = tracklet['poses_dict'][pose_idx]

    t = [pose['tx'], pose['ty'], pose['tz']]
    rz = wrapToPi(pose['rz'])
    corners_3D, orientation_3D = get_corners_and_orientation(corners=corners, rz=rz, l=l, t=t,
                                                             veloToCam=veloToCam, cam=cam)
    if any(corners_3D[2, :] < 0.5) or any(orientation_3D[2, :] < 0.5):
        return False

    corners_2D = projectToImage(corners_3D, K)
    box = {'x1': min(corners_2D[0, :]),
           'x2': max(corners_2D[0, :]),
           'y1': min(corners_2D[1, :]),
           'y2': max(corners_2D[1, :])}

    # checking for not seen box in 2D projection. It is not included in original matlab scipt. Fuck you, matlab, just fuck you.
    if box['x1'] > image_resolution[0] or box['x2'] < 0 or box['y1'] > image_resolution[1] or box['y2'] < 0:
        return False

    return True
