import numpy as np
from python.data_utils import tracklet_to_bounding_box
from math import pi


class DatasetFilterer:

    def __init__(self, min_distance: int, max_distance: int, treshold_degrees:int = None):
        self.treshold_degrees = treshold_degrees
        self.max_distance = max_distance
        self.min_distance = min_distance

    def is_for_dataset(self, tracklet, frame, calib_dir, cam):
        if tracklet['objectType'] != 'Car':
            return False

        pose_idx = frame - tracklet['first_frame']
        pose = tracklet['poses_dict'][pose_idx]

        # filter out occluded tracklets
        if pose['occlusion'] != 0:
            return False

        # filter out cars with high rotation

        corners, t, rz, box, corners_3D, pose_idx, orientation_3D = tracklet_to_bounding_box(tracklet=tracklet,
                                                                                             cam=cam,
                                                                                             frame=frame,
                                                                                             calib_dir=calib_dir)

        r = np.linalg.norm((corners_3D[0, :], corners_3D[2, :]), axis=0)  # r is used for distance measurement

        # Rotation is calculated in 3D coordinates.
        # Orientation of bounding box is transferred to cylindrical coordinates (so we do not care about the Y axis). [x, z => r, theta, y => y]
        # Angle in cylindrical coordinates is ange under which car is seen, so it is used for filtering.
        # orientation_3D is represented by 2 points.
        # car is in front of camera when angle of both points (in cylindrical coordinates) is same
        if self.treshold_degrees is not None:
            orientation_vector = orientation_3D[:, 1] - orientation_3D[:, 0]
            vector_theta = np.arctan2(orientation_vector[2], orientation_vector[0])
            start_theta = np.arctan2(orientation_3D[2, 0], orientation_3D[0, 0])
            angle = vector_theta - start_theta
            treshold = self.treshold_degrees * pi / 180
            if abs(angle) > treshold:
                return False

        # instead of fixed distance in X axis, we use distance from cylindrical coordinates, because this is more accurate
        distance = r[7]
        # corner_ldf = corners_3D[:, 7]
        # distance = corner_ldf.T[2]
        if distance < self.min_distance or distance > self.max_distance:
            return False

        return True
