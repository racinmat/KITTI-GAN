import matplotlib
# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

import atexit
import diskcache as diskcache
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
from devkit.python.utils import loadFromFile, load_image, timeit, transform_to_range
from devkit.python.wrapToPi import wrapToPi
from math import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from functools import lru_cache
import os
import diskcache
from utils import get_P_velo_to_img, tracklet_to_bounding_box, is_tracklet_seen


class Cache(diskcache.Cache):
    def memoize(self, func):

        def wrapper(*args, **kw):
            key = (args, frozenset(kw.items()))
            try:
                return self[key]
            except KeyError:
                value = func(*args, **kw)
                self[key] = value
                return value

        return wrapper


cache_velo = Cache('./cache/velo')
cache_bb = Cache('./cache/bb')

atexit.register(lambda: cache_velo.close())
atexit.register(lambda: cache_bb.close())


@lru_cache(maxsize=32)
def load_tracklets(base_dir):
    # read tracklets for the selected sequence
    tracklets = readTracklets(base_dir + '/tracklet_labels.xml')
    return tracklets


def is_for_dataset(tracklet, frame, calib_dir, cam):
    # only cars in dataset
    if tracklet['objectType'] != 'Car':
        return False

    pose_idx = frame - tracklet['first_frame']
    pose = tracklet['poses_dict'][pose_idx]

    # filter out occluded tracklets
    if pose['occlusion'] != 0:
        return False

    treshold_degrees = 5
    treshold = treshold_degrees * pi / 180
    # filter out cars with high rotation
    if pose['rz'] > treshold or pose['rz'] < - treshold:
        return False

    corners, t, rz, box, corners_3D = tracklet_to_bounding_box(tracklet=tracklet,
                                                               cam=cam,
                                                               frame=frame,
                                                               calib_dir=calib_dir)
    corner_ldf = corners_3D[:, 7]
    distance = corner_ldf.T[2]
    min_distance = 5
    max_distance = 50
    if distance < min_distance or distance > max_distance:
        return False

    return True


# @timeit
@cache_velo.memoize
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


# @timeit
def pointcloud_to_image(velo, velo_img, img=None, grayscale=False):
    image_resolution = np.array([1242, 375])
    fig = plt.figure()
    plt.axes([0, 0, 1, 1])

    # plot points

    if grayscale:
        # transform to grayscale color, black is nearest (lowest distance -> lowest value)
        colors = transform_to_range(5, 80, 0, 1, velo[:, 0])
        plt.style.use('grayscale')
        plt.scatter(x=velo_img[:, 0], y=velo_img[:, 1], c=colors, marker='o', s=1)
    else:
        cols = matplotlib.cm.jet(np.arange(256))  # jet is colormap, represented by lookup table
        # because I want the most distant value to have more cold color (lower value)
        col_indices = np.round(transform_to_range(1/80, 1/5, 0, 255, 1 / velo[:, 0])).astype(int)
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


# @timeit
def get_x_y_data_for(tracklet, frame, cam, calib_dir, current_dir, with_image=False, grayscale=False):
    image_resolution = np.array([1242, 375])

    corners, t, rz, box, corners_3D = tracklet_to_bounding_box(tracklet=tracklet,
                                                   cam=cam,
                                                   frame=frame,
                                                   calib_dir=calib_dir)

    area = (box['x1'], box['y1'], box['x2'], box['y2'])
    original_area = (0, 0, image_resolution[0], image_resolution[1])
    # because some parts of areas are out of the image
    area = (
        max(area[0], original_area[0]),
        max(area[1], original_area[1]),
        min(area[2], original_area[2]),
        min(area[3], original_area[3]),
    )

    velo, velo_img = get_pointcloud(current_dir, frame, calib_dir, cam, area=area)

    if with_image:
        img = load_image('{:s}/image_{:02d}/data/{:010d}.png'.format(current_dir, cam, frame))
    else:
        img = None

    buf, im = pointcloud_to_image(velo, velo_img, img, grayscale)
    if grayscale:
        im = im.convert('L')
    cropped_im = im.crop(area)
    # cropped_im.save('images/{:d}.{:s}.png'.format(frame, str(area)), format='png')
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


# @timeit
def main():
    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]
    drive_dir = './data/2011_09_26/2011_09_26_'
    calib_dir = './data/2011_09_26'

    cam = 2

    data = []

    for i, drive in enumerate(drives):
        current_dir = drive_dir + drive
        image_dir = current_dir + '/image_{:02d}/data'.format(cam)
        # get number of images for this dataset
        frames = len(glob.glob(image_dir + '/*.png'))
        # start = 18
        # end = 20
        start = 0
        end = frames
        # end = round(frames / 50)

        print('processing drive no. {:d}/{:d} with {:d} frames'.format(i + 1, len(drives), frames))

        tracklets = load_tracklets(base_dir=current_dir)
        for frame in range(start, end):
            # percentage printing
            percent = 5
            part = int(((100 * frame) / frames) / percent)
            previous = int(((100 * (frame - 1)) / frames) / percent)
            if part - previous > 0:
                print(str(percent * part) + '% extracted.')

            if not velodyne_data_exist(current_dir, frame):
                continue

            for j, tracklet in enumerate(tracklets):
                if not is_tracklet_seen(tracklet=tracklet, frame=frame, calib_dir=calib_dir, cam=cam):
                    continue

                if not is_for_dataset(tracklet=tracklet, frame=frame, cam=cam, calib_dir=calib_dir):
                    continue

                pair = get_x_y_data_for(tracklet=tracklet,
                                        frame=frame,
                                        cam=cam,
                                        calib_dir=calib_dir,
                                        current_dir=current_dir,
                                        with_image=False,
                                        grayscale=True)
                data.append(pair)

        file = open('data/extracted/tracklets_points_grayscale_bg_white_' + drive + '.data', 'wb')
        pickle.dump(data, file)
        file.close()


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

    pair = get_x_y_data_for(tracklet=tracklet,
                            frame=frame,
                            cam=cam,
                            calib_dir=calib_dir,
                            current_dir=current_dir,
                            with_image=False,
                            grayscale=False)
    im = Image.fromarray(pair['y'])
    im.save('image-bg-jet.png')

    pair = get_x_y_data_for(tracklet=tracklet,
                            frame=frame,
                            cam=cam,
                            calib_dir=calib_dir,
                            current_dir=current_dir,
                            with_image=False,
                            grayscale=True)
    im = Image.fromarray(pair['y'])
    im.save('image-bg-grayscale.png')


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
