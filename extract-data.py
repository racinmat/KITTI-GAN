import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

import atexit
import glob
from PIL import Image
from devkit.python.readTracklets import read_tracklets
import numpy as np
from devkit.python.utils import load_image, Timeit
from math import pi
import pickle
import os
from utils import tracklet_to_bounding_box, is_tracklet_seen, Cache, get_pointcloud, pointcloud_to_image
from devkit.python.utils import timeit


# cache_bb = Cache('./cache/bb')
#
# atexit.register(lambda: cache_bb.close())


# @lru_cache(maxsize=32)
def load_tracklets(base_dir):
    # read tracklets for the selected sequence
    tracklets = read_tracklets(base_dir + '/tracklet_labels.xml')
    return tracklets


def is_for_dataset(tracklet, frame, calib_dir, cam):
    min_distance = 15
    max_distance = 45
    treshold_degrees = 5

    # only cars in dataset
    if tracklet['objectType'] != 'Car':
        return False

    pose_idx = frame - tracklet['first_frame']
    pose = tracklet['poses_dict'][pose_idx]

    # filter out occluded tracklets
    if pose['occlusion'] != 0:
        return False

    treshold = treshold_degrees * pi / 180
    # filter out cars with high rotation

    corners, t, rz, box, corners_3D, pose_idx, orientation_3D = tracklet_to_bounding_box(tracklet=tracklet,
                                                                                         cam=cam,
                                                                                         frame=frame,
                                                                                         calib_dir=calib_dir)

    # Rotation is calculated in 3D coordinates.
    # Orientation of bounding box is transferred to cylindrical coordinates (so we do not care about the Y axis). [x, z => r, theta, y => y]
    # Angle in cylindrical coordinates is ange under which car is seen, so it is used for filtering.
    # orientation_3D is represented by 2 points.
    # car is in front of camera when angle of both points (in cylindrical coordinates) is same
    orientation_vector = orientation_3D[:, 1] - orientation_3D[:, 0]
    vector_theta = np.arctan2(orientation_vector[2], orientation_vector[0])
    start_theta = np.arctan2(orientation_3D[2, 0], orientation_3D[0, 0])
    r = np.linalg.norm((corners_3D[0, :], corners_3D[2, :]), axis=0)  # r is used for distance measurement
    angle = vector_theta - start_theta
    if angle > treshold or angle < - treshold:
        return False

    # instead of fixed distance in X axis, we use distance from cylindrical coordinates, because this is more accurate
    distance = r[7]
    # corner_ldf = corners_3D[:, 7]
    # distance = corner_ldf.T[2]
    if distance < min_distance or distance > max_distance:
        return False

    return True


def velodyne_data_exist(base_dir, frame):
    filename = '{:s}/velodyne_points/data/{:010d}.bin'.format(base_dir, frame)
    return os.path.isfile(filename)


@Timeit
def get_x_y_data_for(tracklet, frame, cam, calib_dir, current_dir, with_image=False, grayscale=False):
    image_resolution = np.array([1242, 375])

    corners, t, rz, box, corners_3D, pose_idx, orientation_3D = tracklet_to_bounding_box(tracklet=tracklet,
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

    corner_ldf = corners_3D[:, 7]
    return {
        'x': [
            rz,
            tracklet['h'] / tracklet['w'],  # height/width ratio
            tracklet['l'] / tracklet['w'],  # depth/width ratio
            corner_ldf.T[2]  # distance from image
        ],
        'y': pix,
        'metadata': {
            'tracklet': tracklet,
            'frame': frame,
            'pose_idx': pose_idx,
            'velo': velo,
            'velo_img': velo_img
        }
    }


@timeit
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

    for i, drive in enumerate(drives):
        data = []
        current_dir = drive_dir + drive
        image_dir = current_dir + '/image_{:02d}/data'.format(cam)
        # get number of images for this dataset
        frames = len(glob.glob(image_dir + '/*.png'))
        # start = 0
        # end = 40
        start = 0
        end = frames
        # end = round(frames / 50)

        print('processing drive no. {:d}/{:d} with {:d} frames'.format(i + 1, len(drives), end - start))

        length = end - start
        tracklets = load_tracklets(base_dir=current_dir)
        for frame in range(start, end):
            # percentage printing
            percent = 5
            part = int(((100 * (frame - start)) / length) / percent)
            previous = int(((100 * (frame - start - 1)) / length) / percent)
            if part - previous > 0:
                print(str(percent * part) + '% extracted.')

            if not velodyne_data_exist(current_dir, frame):
                continue

            for j, tracklet in enumerate(tracklets):
                if not is_tracklet_seen(tracklet=tracklet, frame=frame, calib_dir=calib_dir, cam=cam):
                    continue

                if not is_for_dataset(tracklet=tracklet, frame=frame, cam=cam, calib_dir=calib_dir):
                    continue

                sample = get_x_y_data_for(tracklet=tracklet,
                                          frame=frame,
                                          cam=cam,
                                          calib_dir=calib_dir,
                                          current_dir=current_dir,
                                          with_image=False,
                                          grayscale=True)

                # visualization of sample
                # buf, im = sample_to_image(sample, cam, calib_dir, current_dir)
                # im.save('images/extraction/' + drive + '_{:d}_src_frame_{:d}.png'.format(j, frame))
                # buf.close()
                # end of visualization

                data.append(sample)

        file_name = 'data/extracted/tracklets_points_grayscale_bg_white_' + drive
        if start != 0 or end != frames:
            file_name = file_name + "_{:d}_{:d}".format(start, end)
        file_name = file_name + '.data'
        file = open(file_name, 'wb')
        pickle.dump(data, file)
        print('data saved to file: {}, extracted {} samples.'.format(file_name, len(data)))
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
    print("extraction done")
    # print(tracklet_to_bounding_box.get_time())
    # print(get_pointcloud.get_time())
    # print(pointcloud_to_image.get_time())
    # print(is_tracklet_seen.get_time())
    # print(get_x_y_data_for.get_time())

    # print(load_tracklets.cache_info())
    # print(load_calibration_rigid.cache_info())
    # print(load_calibration.cache_info())
    # print(load_calibration_cam_to_cam.cache_info())
    # print(load_image.cache_info())
    # print(read_tracklets.cache_info())
    # print(loadFromFile.cache_info())
    # print(get_corners.cache_info())
    # print(get_P_velo_to_img.cache_info())
