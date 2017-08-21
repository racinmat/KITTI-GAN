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
from devkit.python.load_calibration import load_calibration
from devkit.python.readTracklets import read_tracklets

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


# for i, drive in enumerate(drives):
#     current_dir = drive_dir + drive
#     tracklets = read_tracklets(current_dir + '/tracklet_labels.xml')
#     tracklets2 = read_tracklets(current_dir + '/tracklet_labels.xml')
#     tracklets_cached = read_tracklets_cached(current_dir + '/tracklet_labels.xml')
#     print(tracklets_equal(tracklets, tracklets2))
#     print(tracklets_equal(tracklets, tracklets_cached))
#     pass


# velo_to_camera = [
#     [2.34773698e-04, -9.99944155e-01, -1.05634778e-02, 5.93721868e-02],
#     [0.01044941, 0.01056535, -0.99988957, -0.07510879],
#     [9.99945389e-01, 1.24365378e-04, 1.04513030e-02, -2.72132796e-01],
#     [0., 0., 0., 1.],
# ]
# 
# trans_1 = [
#     [2, 1],
#     [1, 2],
# ]
# 
# vec_1 = [1, 0]
# vec_2 = [0, 1]
# # np.dot()

t_1 = [[-21.88474084, -20.2576105, -20.37183918, -21.99896953, -21.9028679, -20.27573756, -20.38996624, -22.01709658],
       [1.92701447, 1.90927746, 1.87348452, 1.89122153, 0.21119176, 0.19345476, 0.15766181, 0.17539882],
       [48.2920398, 48.23971212, 44.69990092, 44.7522286, 48.30997436, 48.25764668, 44.71783548, 44.77016316],
       [1., 1., 1., 1., 1., 1., 1., 1.]]

t_2 = [[-22.21369679, -20.7107318, -20.61106783, -22.11403282, -22.22976088, -20.72679589, -20.62713192, -22.13009691],
       [1.81266215, 1.79723208, 1.75952516, 1.77495524, 0.29211008, 0.27668, 0.23897309, 0.25440316],
       [59.30642015, 59.34930143, 55.8425665, 55.79968522, 59.32231365, 59.36519493, 55.85846, 55.81557872],
       [1., 1., 1., 1., 1., 1., 1., 1.]]

t_3 = [[-1.71966301e+00, -9.12877793e-02, 1.55816821e-02, -1.61279354e+00, -1.73679688e+00, -1.08421659e-01,
        -1.55219754e-03, -1.62992742e+00],
       [1.71182884, 1.69503187, 1.64688273, 1.6636797, 0.09001565, 0.07321868, 0.02506954, 0.04186651],
       [37.03159974, 37.07046457, 32.57199137, 32.53312655, 37.04855168, 37.0874165, 32.5889433, 32.55007848],
       [1., 1., 1., 1., 1., 1., 1., 1.]]

print(t_1)
load_calibration(calib_dir)
