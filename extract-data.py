from devkit.python.loadCalibration import loadCalibration
from devkit.python.projectToImage import projectToImage
from devkit.python.readTracklets import readTracklets
import numpy as np
import glob
from devkit.python.wrapToPi import wrapToPi
from math import cos, sin


def load_tracklets(base_dir):
    # read tracklets for the selected sequence
    tracklets = readTracklets(base_dir + '/tracklet_labels.xml')
    return tracklets


def is_tracklet_seen(tracklet, frame, veloToCam, cam):
    pose_idx = frame - tracklet['first_frame']
    corners = get_corners(tracklet)
    # only draw tracklets that are visible in current frame
    if pose_idx < 0 or pose_idx > (np.size(tracklet['poses'], 1) - 1):
        return False

    rz = wrapToPi(tracklet['poses'][5, :])
    t = np.vstack((tracklet['poses'][0, :], tracklet['poses'][1, :], tracklet['poses'][2, :]))
    l = tracklet['l']
    corners_3D, orientation_3D = get_corners_and_orientation(corners=corners, rz=rz, pose_idx=pose_idx, l=l, t=t,
                                                             veloToCam=veloToCam, cam=cam)
    if any(corners_3D[2, :] < 0.5) or any(orientation_3D[2, :] < 0.5):
        return False

    return True


def tracklet_to_bounding_box(tracklet, cam, frame, veloToCam, K):
    corners = get_corners(tracklet)
    t = np.vstack((tracklet['poses'][0, :], tracklet['poses'][1, :], tracklet['poses'][2, :]))
    rz = wrapToPi(tracklet['poses'][5, :])
    occlusion = tracklet['poses'][7, :]

    pose_idx = frame - tracklet['first_frame']
    l = tracklet['l']
    corners_3D, orientation_3D = get_corners_and_orientation(corners=corners, rz=rz, pose_idx=pose_idx, l=l, t=t,
                                                             veloToCam=veloToCam, cam=cam)
    corners_2D = projectToImage(corners_3D, K)
    orientation_2D = projectToImage(orientation_3D, K)

    return corners, t, rz, occlusion, corners_3D, orientation_3D, corners_2D, orientation_2D


def rz_to_R(rz):
    R = [[cos(rz), - sin(rz), 0],
         [sin(rz), cos(rz), 0],
         [0, 0, 1]]
    return R


def get_corners(tracklet):
    w = tracklet['w']
    h = tracklet['h']
    l = tracklet['l']
    corners = {
        'x': [l / 2, l / 2, - l / 2, - l / 2, l / 2, l / 2, - l / 2, - l / 2],
        'y': [w / 2, - w / 2, - w / 2, w / 2, w / 2, - w / 2, - w / 2, w / 2],
        'z': [0, 0, 0, 0, h, h, h, h]
    }
    return corners


def get_corners_and_orientation(corners, rz, pose_idx, l, t, veloToCam, cam):
    R = rz_to_R(rz[pose_idx])
    corners_3D = np.dot(R, [corners['x'], corners['y'], corners['z']])
    corners_3D[0, :] = corners_3D[0, :] + t[0, pose_idx]
    corners_3D[1, :] = corners_3D[1, :] + t[1, pose_idx]
    corners_3D[2, :] = corners_3D[2, :] + t[2, pose_idx]
    corners_3D = np.dot(veloToCam[cam], np.vstack((corners_3D, np.ones((1, np.size(corners_3D, 1))))))
    orientation_3D = np.dot(R, [[0.0, 0.7 * l], [0.0, 0.0], [0.0, 0.0]])
    orientation_3D[0, :] = orientation_3D[0, :] + t[0, pose_idx]
    orientation_3D[1, :] = orientation_3D[1, :] + t[1, pose_idx]
    orientation_3D[2, :] = orientation_3D[2, :] + t[2, pose_idx]
    orientation_3D = np.dot(veloToCam[cam], np.vstack((orientation_3D, np.ones((1, np.size(orientation_3D, 1))))))
    return corners_3D, orientation_3D


def main():
    dirs = [
        './data/2011_09_26/2011_09_26_drive_0009_sync',
        './data/2011_09_26/2011_09_26_drive_0015_sync',
        './data/2011_09_26/2011_09_26_drive_0023_sync',
        './data/2011_09_26/2011_09_26_drive_0032_sync',
    ]
    calib_dir = './data/2011_09_26'

    cam = 2
    frame = 20

    veloToCam, K = loadCalibration(calib_dir)

    for dir in dirs:
        tracklets = load_tracklets(base_dir=dir)
        for tracklet in tracklets:
            if is_tracklet_seen(tracklet=tracklet, frame=frame, veloToCam=veloToCam, cam=cam):
                corners, t, rz, occlusion, corners_3D, orientation_3D, corners_2D, orientation_2D = tracklet_to_bounding_box(tracklet, cam=cam, frame=frame, veloToCam=veloToCam, K=K)


if __name__ == '__main__':
    main()

