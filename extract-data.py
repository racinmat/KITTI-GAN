from devkit.python.loadCalibration import loadCalibration
from devkit.python.projectToImage import projectToImage
from devkit.python.readTracklets import readTracklets
import numpy as np
from smop.core import *
import glob
from devkit.python.wrapToPi import wrapToPi
from math import cos, sin


def load_tracklets(base_dir=None, calib_dir=None):

    if base_dir is None:
        base_dir = './../../data/2011_09_26/2011_09_26_drive_0009_sync'

    if calib_dir is None:
        calib_dir = './../../data/2011_09_26'

    cam = 2

    # get image sub-directory
    image_dir = base_dir + '/image_{:02d}/data'.format(cam)
    # get number of images for this dataset
    nimages = len(glob.glob(image_dir + '/*.png'))
    # read calibration for the day
    veloToCam, K = loadCalibration(calib_dir)
    # read tracklets for the selected sequence
    tracklets = readTracklets(base_dir + '/tracklet_labels.xml')

    # extract tracklets
    # LOCAL OBJECT COORDINATE SYSTEM:
    #   x -> facing right
    #   y -> facing forward
    #   z -> facing up
    corners = np.empty_like(tracklets, dtype=dict)
    t = np.empty_like(tracklets, dtype=list)
    rz = np.empty_like(tracklets, dtype=list)
    occlusion = np.empty_like(tracklets, dtype=list)
    for it, tracklet in enumerate(tracklets):
        # shortcut for tracklet dimensions
        w = tracklet['w']
        h = tracklet['h']
        l = tracklet['l']
        corners[it] = {}
        corners[it]['x'] = [l / 2, l / 2, - l / 2, - l / 2, l / 2, l / 2, - l / 2, - l / 2]
        corners[it]['y'] = [w / 2, - w / 2, - w / 2, w / 2, w / 2, - w / 2, - w / 2, w / 2]
        corners[it]['z'] = [0, 0, 0, 0, h, h, h, h]
        t[it] = np.vstack((tracklet['poses'][0, :], tracklet['poses'][1, :], tracklet['poses'][2, :]))
        rz[it] = wrapToPi(tracklet['poses'][5, :])
        occlusion[it] = tracklet['poses'][7, :]

    # 3D bounding box faces (indices for corners)
    face_idx = np.array([[1, 2, 6, 5],  # front face
                         [2, 3, 7, 6],  # left face
                         [3, 4, 8, 7],  # back face
                         [4, 1, 5, 8]]) # right face

    face_idx -= 1 # transformation to 0-based indexation

    # image index
    img_idx = 0

    for it in range(len(tracklets)):
        # get relative tracklet frame index (starting at 0 with first appearance;
        # xml data stores poses relative to the first frame where the tracklet appeared)
        pose_idx = img_idx - tracklets[it]['first_frame']
        # only draw tracklets that are visible in current frame
        if pose_idx < 0 or pose_idx > (size(tracklets[it]['poses'], 2) - 1):
            continue
            # compute 3d object rotation in velodyne coordinates
            # VELODYNE COORDINATE SYSTEM:
            #   x -> facing forward
            #   y -> facing left
            #   z -> facing up
        l = tracklets[it]['l']
        R = [[cos(rz[it][pose_idx]), - sin(rz[it][pose_idx]), 0], [sin(rz[it][pose_idx]), cos(rz[it][pose_idx]), 0],
             [0, 0, 1]]
        corners_3D = dot(R, [corners[it]['x'], corners[it]['y'], corners[it]['z']])
        corners_3D[0, :] = corners_3D[0, :] + t[it][0, pose_idx]
        corners_3D[1, :] = corners_3D[1, :] + t[it][1, pose_idx]
        corners_3D[2, :] = corners_3D[2, :] + t[it][2, pose_idx]
        corners_3D = dot(veloToCam[cam], np.vstack((corners_3D, np.ones((1, size(corners_3D, 2))))))
        orientation_3D = dot(R, [[0.0, 0.7 * l], [0.0, 0.0], [0.0, 0.0]])
        orientation_3D[0, :] = orientation_3D[0, :] + t[it][0, pose_idx]
        orientation_3D[1, :] = orientation_3D[1, :] + t[it][1, pose_idx]
        orientation_3D[2, :] = orientation_3D[2, :] + t[it][2, pose_idx]
        orientation_3D = dot(veloToCam[cam], np.vstack((orientation_3D, np.ones((1, size(orientation_3D, 2))))))
        if any(corners_3D[2, :] < 0.5) or any(orientation_3D[2, :] < 0.5):
            continue
        # project the 3D bounding box into the image plane
        corners_2D = projectToImage(corners_3D, K)
        orientation_2D = projectToImage(orientation_3D, K)
        # compute and draw the 2D bounding box from the 3D box projection
        box = {'x1': min(corners_2D[0, :]),
               'x2': max(corners_2D[0, :]),
               'y1': min(corners_2D[1, :]),
               'y2': max(corners_2D[1, :])}

if __name__ == '__main__':
    dirs = [
        './data/2011_09_26/2011_09_26_drive_0009_sync',
        './data/2011_09_26/2011_09_26_drive_0015_sync',
        './data/2011_09_26/2011_09_26_drive_0023_sync',
        './data/2011_09_26/2011_09_26_drive_0032_sync',
    ]
    calib_dir = './data/2011_09_26'

    for dir in dirs:
        load_tracklets(base_dir=dir, calib_dir=calib_dir)
