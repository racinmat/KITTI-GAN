import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

import glob
import datetime
import io
from PIL import Image
from devkit.python.loadCalibration import loadCalibration
from devkit.python.loadCalibrationCamToCam import loadCalibrationCamToCam
from devkit.python.loadCalibrationRigid import loadCalibrationRigid
from devkit.python.project import project
from devkit.python.projectToImage import projectToImage
from devkit.python.readTracklets import readTracklets
import numpy as np
from devkit.python.utils import loadFromFile, load_image, timeit
from devkit.python.wrapToPi import wrapToPi
from math import cos, sin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from functools import lru_cache
import os


@lru_cache(maxsize=32)
def load_tracklets(base_dir):
    # read tracklets for the selected sequence
    tracklets = readTracklets(base_dir + '/tracklet_labels.xml')
    return tracklets


def is_tracklet_seen(tracklet, frame, calib_dir, cam):
    veloToCam, K = loadCalibration(calib_dir)

    pose_idx = frame - tracklet['first_frame']
    # only draw tracklets that are visible in current frame
    if pose_idx < 0 or pose_idx > (np.size(tracklet['poses'], 1) - 1):
        return False

    corners = get_corners(w=tracklet['w'], h=tracklet['h'], l=tracklet['l'])
    l = tracklet['l']
    pose = tracklet['poses_dict'][pose_idx]

    # filter out occluded tracklets
    if pose['occlusion'] != 0:
        return False

    t = [pose['tx'], pose['ty'], pose['tz']]
    rz = wrapToPi(pose['rz'])
    corners_3D, orientation_3D = get_corners_and_orientation(corners=corners, rz=rz, l=l, t=t,
                                                             veloToCam=veloToCam, cam=cam)
    if any(corners_3D[2, :] < 0.5) or any(orientation_3D[2, :] < 0.5):
        return False

    return True


def tracklet_to_bounding_box(tracklet, cam, frame, veloToCam, K):
    corners = get_corners(w=tracklet['w'], h=tracklet['h'], l=tracklet['l'])
    occlusion = tracklet['poses'][7, :]

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

    # return corners, t, rz, occlusion, corners_3D, orientation_3D, box
    return corners, t, rz, occlusion, box


def rz_to_R(rz):
    R = [[cos(rz), - sin(rz), 0],
         [sin(rz), cos(rz), 0],
         [0, 0, 1]]
    return R


@lru_cache(maxsize=64)
def get_corners(w, h, l):
    corners = {
        'x': [l / 2, l / 2, - l / 2, - l / 2, l / 2, l / 2, - l / 2, - l / 2],
        'y': [w / 2, - w / 2, - w / 2, w / 2, w / 2, - w / 2, - w / 2, w / 2],
        'z': [0, 0, 0, 0, h, h, h, h]
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


@timeit
def get_pointcloud(base_dir, frame, calib_dir, cam, area=None):
    P_velo_to_img = get_P_velo_to_img(calib_dir=calib_dir, cam=cam)
    # load velodyne points
    fname = '{:s}/velodyne_points/data/{:010d}.bin'.format(base_dir, frame)
    velo = loadFromFile(fname, 4, np.float32)
    # keep only every 5-th point for visualization
    # velo = velo[0::5, :]

    # remove all points behind image plane (approximation)
    idx = velo[:, 0] < 5
    velo = velo[np.invert(idx), :]

    # project to image plane (exclude luminance)
    velo_img = project(velo[:, 0:3], P_velo_to_img)

    if area is not None:
        x1, y1, x2, y2 = area
        ll = np.array([x1, y1])  # lower-left
        ur = np.array([x2, y2])  # upper-right

        indices = np.all(np.logical_and(ll <= velo_img, velo_img <= ur), axis=1)
        velo_img = velo_img[indices]
        velo = velo[indices]

    return velo, velo_img


@timeit
def pointcloud_to_image(velo, velo_img, img=None):
    image_resolution = np.array([1242, 375])
    fig = plt.figure()
    plt.axes([0, 0, 1, 1])

    # plot points
    cols = matplotlib.cm.jet(np.arange(256))  # jet is colormap, represented by lookup table
    col_indices = np.round(256 * 5 / velo[:, 0]).astype(int) - 1
    plt.scatter(x=velo_img[:, 0], y=velo_img[:, 1], c=cols[col_indices, 0:3], marker='o', s=1)

    dpi = fig.dpi
    fig.set_size_inches(image_resolution / dpi)
    ax = plt.gca()
    ax.set_xlim((-0.5, image_resolution[0] - 0.5))
    ax.set_ylim((image_resolution[1] - 0.5, -0.5))

    if img is not None:
        plt.imshow(img)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    return buf, im


def velodyne_data_exist(base_dir, frame):
    filename = '{:s}/velodyne_points/data/{:010d}.bin'.format(base_dir, frame)
    return os.path.isfile(filename)


@timeit
def get_x_y_data_for_(tracklet, frame, cam, calib_dir, current_dir, with_image):
    veloToCam, K = loadCalibration(dir=calib_dir)
    image_resolution = np.array([1242, 375])

    corners, t, rz, occlusion, box = tracklet_to_bounding_box(tracklet=tracklet,
                                                              cam=cam,
                                                              frame=frame,
                                                              veloToCam=veloToCam,
                                                              K=K)

    area = (box['x1'], box['y1'], box['x2'], box['y2'])
    original_area = (0, 0, image_resolution[0], image_resolution[1])
    # because some parts of areas are out of the image
    area = (max(area[0], original_area[0]),
            max(area[1], original_area[1]),
            min(area[2], original_area[2]),
            min(area[3], original_area[3]),
            )


    velo, velo_img = get_pointcloud(current_dir, frame, calib_dir, cam, area=area)

    if with_image:
        img = load_image('{:s}/image_{:02d}/data/{:010d}.png'.format(current_dir, cam, frame))
    else:
        img = None

    buf, im = pointcloud_to_image(velo, velo_img, img)
    cropped_im = im.crop(area)
    # cropped_im.save('images/{:d}.{:d}.png'.format(i, j), format='png')
    pix = np.array(cropped_im)
    buf.close()
    return {
        'x': [
            rz,
            tracklet['w'],
            tracklet['h'],
            tracklet['l'],
        ],
        'y': pix
    }


@timeit
def main():
    drives = [
        'drive_0009_sync',
        # 'drive_0015_sync',
        # 'drive_0023_sync',
        # 'drive_0032_sync',
    ]
    drive_dir = './data/2011_09_26/2011_09_26_'
    calib_dir = './data/2011_09_26'

    cam = 2

    # posesData = np.empty((15, 0), dtype=float)
    # for i, drive in enumerate(drives):
    #     dir = drive_dir + drive
    #     tracklets = load_tracklets(base_dir=dir)
    #     for j, tracklet in enumerate(tracklets):
    #         posesData = np.concatenate((posesData, tracklet['poses']), axis=1)
    #
    # nbins = 50
    # fig = plt.figure()
    # for i in range(0, 15):
    #     fig.add_subplot(15, 1, i+1)
    #     plt.hist(x=posesData[i, :], bins=nbins)
    #     plt.title('hist of ' + str(i))
    #
    # fig.set_figheight(20)
    # fig.subplots_adjust(hspace=2)
    # plt.savefig('hists.png')
    # return

    data = []

    start = datetime.datetime.now()

    for i, drive in enumerate(drives):
        current_dir = drive_dir + drive
        image_dir = current_dir + '/image_{:02d}/data'.format(cam)
        # get number of images for this dataset
        frames = len(glob.glob(image_dir + '/*.png'))
        frames = 10

        print('processing drive no. {:d}/{:d} with {:d} frames'.format(i + 1, len(drives), frames))

        tracklets = load_tracklets(base_dir=current_dir)
        for frame in range(frames):
            # percentage printing
            # percent = 5
            # part = int(((100 * frame) / frames) / percent)
            # previous = int(((100 * (frame - 1)) / frames) / percent)
            # if part - previous > 0:
            #     print(str(percent * part) + '% extracted.')
            if not velodyne_data_exist(current_dir, frame):
                continue

            for j, tracklet in enumerate(tracklets):
                if not is_tracklet_seen(tracklet=tracklet, frame=frame, calib_dir=calib_dir, cam=cam):
                    continue
                pair = get_x_y_data_for_(tracklet=tracklet,
                                         frame=frame,
                                         cam=cam,
                                         calib_dir=calib_dir,
                                         current_dir=current_dir,
                                         with_image=False)
                data.append(pair)

        file = open('data/extracted/tracklets_points_image_bg_' + drive + '.data', 'wb')
        pickle.dump(data, file)
        file.close()

    end = datetime.datetime.now()
    diff = end - start
    print(diff)


def extract_one_tracklet():
    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]
    drive_dir = './data/2011_09_26/2011_09_26_'
    calib_dir = './data/2011_09_26'
    cam = 2
    drive = drives[0]
    current_dir = drive_dir + drive
    frame = 0

    tracklets = load_tracklets(base_dir=current_dir)
    tracklet = tracklets[0]
    # pair = get_x_y_data_for_(tracklet=tracklet,
    #                          frame=frame,
    #                          cam=cam,
    #                          calib_dir=calib_dir,
    #                          current_dir=current_dir,
    #                          with_image=False)
    # im = Image.fromarray(pair['y'])
    # im.save('image-white.png')

    pair = get_x_y_data_for_(tracklet=tracklet,
                             frame=frame,
                             cam=cam,
                             calib_dir=calib_dir,
                             current_dir=current_dir,
                             with_image=True)
    im = Image.fromarray(pair['y'])
    im.save('image-bg-subset.png')


if __name__ == '__main__':
    # extract_one_tracklet()
    main()
    # print(load_tracklets.cache_info())
    # print(loadCalibrationRigid.cache_info())
    # print(loadCalibration.cache_info())
    # print(loadCalibrationCamToCam.cache_info())
    # print(load_image.cache_info())
    # print(readTracklets.cache_info())
    # print(loadFromFile.cache_info())
    # print(get_corners.cache_info())
    # print(get_P_velo_to_img.cache_info())
