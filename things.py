import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# numlines = 128
# cols = matplotlib.cm.jet(np.arange(numlines))
#
# for i in np.linspace(0, numlines - 1, numlines):
#     col_idx = int(i)
#     plt.plot(np.arange(numlines), np.tile([i], numlines), linewidth=4, markersize=1, color=cols[col_idx, 0:3])
#
# plt.show()

#
# data = np.random.rand(100, 1)
# # the histogram of the data
# nbins = 5
# # n, bins, patches = plt.hist(rz, nbins, normed=1, facecolor='green', alpha=0.75)
# n, bins, patches = plt.hist(x=data, bins=nbins)
# # n, bins, patches = plt.hist(rz)
#
# plt.xlabel('yaw angle')
# plt.ylabel('frequency')
# plt.title('frequency of yaw angles of bounding box')
# plt.axis([-1, 2, 0, 50])
# # plt.grid(True)
#
# plt.show()
# # plt.savefig('angles.png')
#
# fig = plt.figure()
#
# Z = np.array([
#     [1, 2, 3, 4, 5],
#     [4, 5, 6, 7, 8],
#     [7, 8, 9, 10, 11]
# ])
#
# fig.add_subplot(4, 1, 1)
# im = plt.imshow(Z, cmap='jet')
# plt.colorbar(im, orientation='horizontal')
#
# Z2 = np.array([
#     [1, 2, 3, 4, 3.5, 2.5, 1.5],
#     [2, 1, 3, 3,   3,   2,   1],
#     [7, 8, 9, 7,   8,  20,  30]
# ])
#
# velo_img = Z2[0:2, :].T
# velo = Z2[2, :].T
#
# minimum = 5
# maximum = 80
# cols = matplotlib.cm.jet(np.arange(256))  # jet is colormap, represented by lookup table
# fig.add_subplot(4, 1, 2)
#
# for i in range(np.size(velo_img, 0)):
#     col_idx = int(round(256 * 5 / velo[i])) - 1
#     plt.plot(velo_img[i, 0], velo_img[i, 1], 'o', linewidth=4, markersize=1, color=cols[col_idx, 0:3])
#
# Z2_i = 1 / Z2[2, :]
# fig.add_subplot(4, 1, 3)
# im = plt.scatter(x=Z2[0, :], y=Z2[1, :], c=Z2_i, cmap='jet', marker='o', s=1)
# plt.colorbar(im, orientation='horizontal')
#
# col_indices = np.round(256 * 5 / velo).astype(int) - 1
# fig.add_subplot(4, 1, 4)
# im = plt.scatter(x=Z2[0, :], y=Z2[1, :], c=cols[col_indices, 0:3], marker='o', s=1)
#
# plt.show()


# fig = plt.figure()
# Z = np.array([
#     [1, 2, 3, 4, 5],
#     [4, 5, 6, 7, 8],
#     [7, 8, 9, 10, 11]
# ])
#
# fig.add_subplot(4, 1, 1)
# im = plt.imshow(Z, cmap='jet')
# plt.colorbar(im, orientation='horizontal')
#
# Z2 = np.array([
#     [1, 2, 3, 4, 3.5, 2.5, 1.5],
#     [2, 1, 3, 3,   3,   2,   1],
#     [7, 8, 9, 7,   8,  20,  30]
# ])
#
# velo_img = Z2[0:2, :].T
# velo = Z2[2, :].T
#
# minimum = 5
# maximum = 80
# cols = matplotlib.cm.jet(np.arange(256))  # jet is colormap, represented by lookup table
# fig.add_subplot(4, 1, 2)
#
# for i in range(np.size(velo_img, 0)):
#     col_idx = int(round(256 * 5 / velo[i])) - 1
#     plt.plot(velo_img[i, 0], velo_img[i, 1], 'o', linewidth=4, markersize=1, color=cols[col_idx, 0:3])
#
# Z2_i = 1 / Z2[2, :]
# fig.add_subplot(4, 1, 3)
# im = plt.scatter(x=Z2[0, :], y=Z2[1, :], c=Z2_i, cmap='jet', marker='o', s=1)
# plt.colorbar(im, orientation='horizontal')
#
# col_indices = np.round(256 * 5 / velo).astype(int) - 1
# fig.add_subplot(4, 1, 4)
# im = plt.scatter(x=Z2[0, :], y=Z2[1, :], c=cols[col_indices, 0:3], marker='o', s=1)
#
# plt.show()
from devkit.python.readTracklets import read_tracklets, read_tracklets_cached

drives = [
    'drive_0009_sync',
    'drive_0015_sync',
    'drive_0023_sync',
    'drive_0032_sync',
]
drive_dir = './data/2011_09_26/2011_09_26_'
calib_dir = './data/2011_09_26'


def tracklets_equal(tracklets_1, tracklets_2):
    if len(tracklets_1) is not len(tracklets_2):
        return False

    for t_1, t_2 in zip(tracklets_1, tracklets_2):
        if t_1['w'] != t_2['w']:
            return False
        if t_1['h'] != t_2['h']:
            return False
        if t_1['l'] != t_2['l']:
            return False
        if t_1['finished'] != t_2['finished']:
            return False
        if t_1['first_frame'] != t_2['first_frame']:
            return False
        if t_1['objectType'] != t_2['objectType']:
            return False
        if len(t_1['poses_dict']) != len(t_2['poses_dict']):
            return False
        for p_1, p_2 in zip(t_1['poses_dict'], t_2['poses_dict']):
            if p_1['amt_border_kf'] != p_2['amt_border_kf']:
                return False
            if p_1['amt_border_l'] != p_2['amt_border_l']:
                return False
            if p_1['amt_border_r'] != p_2['amt_border_r']:
                return False
            if p_1['amt_occlusion'] != p_2['amt_occlusion']:
                return False
            if p_1['amt_occlusion_kf'] != p_2['amt_occlusion_kf']:
                return False
            if p_1['occlusion'] != p_2['occlusion']:
                return False
            if p_1['occlusion_kf'] != p_2['occlusion_kf']:
                return False
            if p_1['rx'] != p_2['rx']:
                return False
            if p_1['ry'] != p_2['ry']:
                return False
            if p_1['rz'] != p_2['rz']:
                return False
            if p_1['tx'] != p_2['tx']:
                return False
            if p_1['ty'] != p_2['ty']:
                return False
            if p_1['tz'] != p_2['tz']:
                return False
            if p_1['state'] != p_2['state']:
                return False
            if p_1['truncation'] != p_2['truncation']:
                return False
        return True


for i, drive in enumerate(drives):
    current_dir = drive_dir + drive
    tracklets = read_tracklets(current_dir + '/tracklet_labels.xml')
    tracklets2 = read_tracklets(current_dir + '/tracklet_labels.xml')
    tracklets_cached = read_tracklets_cached(current_dir + '/tracklet_labels.xml')
    print(tracklets_equal(tracklets, tracklets2))
    print(tracklets_equal(tracklets, tracklets_cached))
    pass