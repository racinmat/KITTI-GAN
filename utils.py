import io
import matplotlib
from devkit.python.drawBox2D import drawBox2D
import atexit
import diskcache
from devkit.python.load_calibration import load_calibration, load_calibration_cam_to_cam, load_calibration_rigid
from functools import lru_cache
from devkit.python.project import project
from devkit.python.utils import loadFromFile, transform_to_range, load_image, Timeit
from devkit.python.wrapToPi import wrapToPi
import numpy as np
from devkit.python.projectToImage import projectToImage
from math import cos, sin
import matplotlib.pyplot as plt
from PIL import Image
from devkit.python.utils import timeit


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
atexit.register(lambda: cache_velo.close())


# @lru_cache(maxsize=64)
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
    corners_3D_cam = np.dot(veloToCam[cam], np.vstack((corners_3D, np.ones((1, np.size(corners_3D, 1))))))
    orientation_3D = np.dot(R, [[0.0, 0.7 * l], [0.0, 0.0], [0.0, 0.0]])
    orientation_3D[0, :] = orientation_3D[0, :] + t[0]
    orientation_3D[1, :] = orientation_3D[1, :] + t[1]
    orientation_3D[2, :] = orientation_3D[2, :] + t[2]
    orientation_3D_cam = np.dot(veloToCam[cam], np.vstack((orientation_3D, np.ones((1, np.size(orientation_3D, 1))))))
    return corners_3D_cam, orientation_3D_cam


@lru_cache(maxsize=32)
def get_P_velo_to_img(calib_dir, cam):
    # load calibration
    calib = load_calibration_cam_to_cam(calib_dir + '/calib_cam_to_cam.txt')
    Tr_velo_to_cam = load_calibration_rigid(calib_dir + '/calib_velo_to_cam.txt')
    # compute projection matrix velodyne->image plane
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[0:3, 0:3] = calib['R_rect'][0]
    P_velo_to_img = np.dot(np.dot(calib['P_rect'][cam], R_cam_to_rect), Tr_velo_to_cam)
    return P_velo_to_img


# @cache_bb.memoize
# @lru_cache(maxsize=32)
@Timeit
def tracklet_to_bounding_box(tracklet, cam, frame, calib_dir):
    veloToCam, K = load_calibration(dir=calib_dir)
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

    return corners, t, rz, box, corners_3D, pose_idx, orientation_3D


def rz_to_R(rz):
    R = [[cos(rz), - sin(rz), 0],
         [sin(rz), cos(rz), 0],
         [0, 0, 1]]
    return R


@Timeit
def is_tracklet_seen(tracklet, frame, calib_dir, cam):
    veloToCam, K = load_calibration(calib_dir)
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


# @cache_velo.memoize
@lru_cache(maxsize=32)
@Timeit
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


# @lru_cache(maxsize=32)
@Timeit
def pointcloud_to_image(velo, velo_img, img=None, grayscale=False):
    fig, ax = pointcloud_to_figure(velo, velo_img, img, grayscale)
    buf, im = figure_to_image(fig)
    return buf, im


@Timeit
def pointcloud_to_figure(velo, velo_img, img=None, grayscale=False):
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
        col_indices = np.round(transform_to_range(1 / 80, 1 / 5, 0, 255, 1 / velo[:, 0])).astype(int)
        plt.scatter(x=velo_img[:, 0], y=velo_img[:, 1], c=cols[col_indices, 0:3], marker='o', s=1)

    dpi = fig.dpi
    fig.set_size_inches(image_resolution / dpi)
    ax = plt.gca()
    ax.set_xlim((-0.5, image_resolution[0] - 0.5))
    ax.set_ylim((image_resolution[1] - 0.5, -0.5))

    if img is not None:
        plt.imshow(img)

    return fig, ax


@Timeit
def figure_to_image(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    return buf, im


def bounding_box_to_image(ax, box, occlusion, object_type):
    drawBox2D(ax, box, occlusion, object_type)


def sample_to_image(sample, cam, calib_dir, current_dir):
    metadata = sample['metadata']
    frame = metadata['frame']
    tracklet = metadata['tracklet']

    corners, t, rz, box, corners_3D, pose_idx, orientation_3D = tracklet_to_bounding_box(tracklet=tracklet,
                                                                                         cam=cam,
                                                                                         frame=frame,
                                                                                         calib_dir=calib_dir)

    kitti_img = load_image('{:s}/image_{:02d}/data/{:010d}.png'.format(current_dir, cam, frame))
    velo = metadata['velo']
    velo_img = metadata['velo_img']
    fig, ax = pointcloud_to_figure(velo, velo_img, kitti_img, False)
    bounding_box_to_image(ax=ax, box=box, occlusion=tracklet['poses_dict'][pose_idx]['occlusion'],
                          object_type=tracklet['objectType'])
    buf, im = figure_to_image(fig)
    return buf, im
