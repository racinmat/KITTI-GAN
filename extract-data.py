import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

import io
from PIL import Image
from devkit.python.loadCalibration import loadCalibration
from devkit.python.loadCalibrationCamToCam import loadCalibrationCamToCam
from devkit.python.loadCalibrationRigid import loadCalibrationRigid
from devkit.python.project import project
from devkit.python.projectToImage import projectToImage
from devkit.python.readTracklets import readTracklets
import numpy as np
from devkit.python.utils import loadFromFile
from devkit.python.wrapToPi import wrapToPi
from math import cos, sin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


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
    box = {'x1': min(corners_2D[0, :]),
           'x2': max(corners_2D[0, :]),
           'y1': min(corners_2D[1, :]),
           'y2': max(corners_2D[1, :])}

    return corners, t, rz, occlusion, corners_3D, orientation_3D, box


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


def get_image_with_pointcloud(base_dir, calib_dir, frame, cam):
    image_resolution = np.array([1242, 375])
    # load calibration
    calib = loadCalibrationCamToCam(calib_dir + '/calib_cam_to_cam.txt')
    Tr_velo_to_cam = loadCalibrationRigid(calib_dir + '/calib_velo_to_cam.txt')
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

    dpi = fig.dpi
    fig.set_size_inches(image_resolution / dpi)
    ax = plt.gca()
    ax.set_xlim((-0.5, image_resolution[0] - 0.5))
    ax.set_ylim((image_resolution[1] - 0.5, -0.5))
    plt.imshow(img)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    return buf, im


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

    posesData = np.empty((15, 0), dtype=float)
    for i, dir in enumerate(dirs):
        tracklets = load_tracklets(base_dir=dir)
        for j, tracklet in enumerate(tracklets):
            posesData = np.concatenate((posesData, tracklet['poses']), axis=1)


    # nbins = 50
    # fig = plt.figure()
    # for i in range(1, 15):
    #     fig.add_subplot(14, 1, i)
    #     plt.hist(x=posesData[i, :], bins=nbins)
    #     plt.title('hist of ' + str(i))
    #
    # fig.set_figheight(20)
    # fig.subplots_adjust(hspace=2)
    # plt.savefig('hists.png')
    # return

    for i, dir in enumerate(dirs):
        tracklets = load_tracklets(base_dir=dir)
        for j, tracklet in enumerate(tracklets):
            if is_tracklet_seen(tracklet=tracklet, frame=frame, veloToCam=veloToCam, cam=cam):
                corners, t, rz, occlusion, corners_3D, orientation_3D, box = tracklet_to_bounding_box(tracklet, cam=cam,
                                                                                                      frame=frame,
                                                                                                      veloToCam=veloToCam,
                                                                                                      K=K)
                buf, im = get_image_with_pointcloud(base_dir=dir, calib_dir=calib_dir, frame=frame, cam=cam)
                area = (box['x1'], box['y1'], box['x2'], box['y2'])
                cropped_im = im.crop(area)
                cropped_im.save('images/{:d}.{:d}.png'.format(i, j), format='png')
                buf.close()
                pix = np.array(cropped_im)
                data.append({
                    'x': [
                        rz,
                        tracklet['w'],
                        tracklet['h'],
                        tracklet['l'],
                    ],
                    'y': pix
                })

    file = open('tracklets_points_white_bg.data', 'wb')
    pickle.dump(data, file)
    file.close()


if __name__ == '__main__':
    main()
