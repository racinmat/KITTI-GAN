import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

from math import cos, sin
from devkit.python.drawBox2D import drawBox2D
from devkit.python.drawBox3D import drawBox3D
from devkit.python.projectToImage import projectToImage
from devkit.python.wrapToPi import wrapToPi
from devkit.python.load_calibration import load_calibration
from devkit.python.readTracklets import read_tracklets
from devkit.python.visualization import visualizationUpdate, visualizationInit
import glob
import matplotlib.pyplot as plt
import numpy as np

def run_demoTracklets(base_dir=None, calib_dir=None):

    # KITTI RAW DATA DEVELOPMENT KIT
    #
    # This tool displays the images and the object labels for the benchmark and
    # provides an entry point for writing your own interface to the data set.
    # Before running this tool, set root_dir to the directory where you have
    # downloaded the dataset. 'root_dir' must contain the subdirectory
    # 'training', which in turn contains 'image_2', 'label_2' and 'calib'.
    # For more information about the data format, please look into readme.txt.

    # Input arguments:
    # base_dir .... absolute path to sequence base directory (ends with _sync)
    # calib_dir ... absolute path to directory that contains calibration files

    # Occlusion Coding:
    #   green:  not occluded
    #   yellow: partly occluded
    #   red:    fully occluded
    #   white:  unknown

    # clear and close everything
    print('======= KITTI DevKit Demo =======')
    # options (modify this to select your sequence)
    # the base_dir must contain:
    #   - the data directories (image_00, image_01, ..)
    #   - the tracklet file (tracklet_labels.xml)
    # the calib directory must contain:
    #   - calib_cam_to_cam.txt
    #   - calib_velo_to_cam.txt
    # cameras:
    #   - 0 = left grayscale
    #   - 1 = right grayscale
    #   - 2 = left color
    #   - 3 = right color

    if base_dir is None:
        base_dir = './../../data/2011_09_26/2011_09_26_drive_0009_sync'

    if calib_dir is None:
        calib_dir = './../../data/2011_09_26'

    cam = 2
    frame = 20

    # get image sub-directory
    image_dir = base_dir + '/image_{:02d}/data'.format(cam)
    # get number of images for this dataset
    nimages = len(glob.glob(image_dir + '/*.png'))
    # set up figure
    gh = visualizationInit(image_dir, frame)
    # read calibration for the day
    veloToCam, K = load_calibration(calib_dir)
    # read tracklets for the selected sequence
    tracklets = read_tracklets(base_dir + '/tracklet_labels.xml')

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
    face_idx -= 1

    # visualization update for next frame
    visualizationUpdate(image_dir, gh, frame, nimages)

    for it in range(len(tracklets)):
        # get relative tracklet frame index (starting at 0 with first appearance;
        # xml data stores poses relative to the first frame where the tracklet appeared)
        pose_idx = frame - tracklets[it]['first_frame']
        # only draw tracklets that are visible in current frame
        if pose_idx < 0 or pose_idx > (np.size(tracklets[it]['poses'], 1) - 1):
            # todo: check it if -1 is correct
            continue
            # compute 3d object rotation in velodyne coordinates
            # VELODYNE COORDINATE SYSTEM:
            #   x -> facing forward
            #   y -> facing left
            #   z -> facing up
        l = tracklets[it]['l']
        R = [[cos(rz[it][pose_idx]), - sin(rz[it][pose_idx]), 0], [sin(rz[it][pose_idx]), cos(rz[it][pose_idx]), 0],
             [0, 0, 1]]
        corners_3D = np.dot(R, [corners[it]['x'], corners[it]['y'], corners[it]['z']])
        corners_3D[0, :] = corners_3D[0, :] + t[it][0, pose_idx]
        corners_3D[1, :] = corners_3D[1, :] + t[it][1, pose_idx]
        corners_3D[2, :] = corners_3D[2, :] + t[it][2, pose_idx]
        corners_3D = np.dot(veloToCam[cam], np.vstack((corners_3D, np.ones((1, np.size(corners_3D, 1))))))
        orientation_3D = np.dot(R, [[0.0, 0.7 * l], [0.0, 0.0], [0.0, 0.0]])
        orientation_3D[0, :] = orientation_3D[0, :] + t[it][0, pose_idx]
        orientation_3D[1, :] = orientation_3D[1, :] + t[it][1, pose_idx]
        orientation_3D[2, :] = orientation_3D[2, :] + t[it][2, pose_idx]
        orientation_3D = np.dot(veloToCam[cam], np.vstack((orientation_3D, np.ones((1, np.size(orientation_3D, 1))))))
        if any(corners_3D[2, :] < 0.5) or any(orientation_3D[2, :] < 0.5):
            continue
        # project the 3D bounding box into the image plane
        corners_2D = projectToImage(corners_3D, K)
        orientation_2D = projectToImage(orientation_3D, K)
        drawBox3D(gh[1], occlusion[it][pose_idx], corners_2D, face_idx, orientation_2D)
        # compute and draw the 2D bounding box from the 3D box projection
        box = {'x1': min(corners_2D[0, :]),
               'x2': max(corners_2D[0, :]),
               'y1': min(corners_2D[1, :]),
               'y2': max(corners_2D[1, :])}
        drawBox2D(gh[0], box, occlusion[it][pose_idx], tracklets[it]['objectType'])

    plt.savefig('bar3.png')

if __name__ == '__main__':
    run_demoTracklets()
