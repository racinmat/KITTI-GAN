from smop.core import *


@function
def run_demoTracklets(base_dir=None, calib_dir=None):
    nargin = run_demoTracklets.nargin

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
    disp('======= KITTI DevKit Demo =======')
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

    # get image sub-directory
    image_dir = fullfile(base_dir, sprintf('/image_%02d/data', cam))
    # get number of images for this dataset
    nimages = length(dir(fullfile(image_dir, '*.png')))
    # set up figure
    gh = visualization('init', image_dir)
    # read calibration for the day
    veloToCam, K = loadCalibration(calib_dir, nargout=2)
    # read tracklets for the selected sequence
    tracklets = readTracklets(cat(base_dir, '/tracklet_labels.xml'))

    # tracklets = readTrackletsMex([base_dir '/tracklet_labels.xml']); # fast version

    # extract tracklets
    # LOCAL OBJECT COORDINATE SYSTEM:
    #   x -> facing right
    #   y -> facing forward
    #   z -> facing up
    for it in arange(1, numel(tracklets)).reshape(-1):
        # shortcut for tracklet dimensions
        w = tracklets[it].w
        # /opt/project/devkit/matlab/run_demoTracklets.m:77
        h = tracklets[it].h
        # /opt/project/devkit/matlab/run_demoTracklets.m:78
        l = tracklets[it].l
        # /opt/project/devkit/matlab/run_demoTracklets.m:79
        corners[it].x = copy(cat(l / 2, l / 2, - l / 2, - l / 2, l / 2, l / 2, - l / 2, - l / 2))
        # /opt/project/devkit/matlab/run_demoTracklets.m:82
        corners[it].y = copy(cat(w / 2, - w / 2, - w / 2, w / 2, w / 2, - w / 2, - w / 2, w / 2))
        # /opt/project/devkit/matlab/run_demoTracklets.m:83
        corners[it].z = copy(cat(0, 0, 0, 0, h, h, h, h))
        # /opt/project/devkit/matlab/run_demoTracklets.m:84
        t[it] = cat([tracklets[it].poses(1, arange())], [tracklets[it].poses(2, arange())],
                    [tracklets[it].poses(3, arange())])
        # /opt/project/devkit/matlab/run_demoTracklets.m:87
        rz[it] = wrapToPi(tracklets[it].poses(6, arange()))
        # /opt/project/devkit/matlab/run_demoTracklets.m:88
        occlusion[it] = tracklets[it].poses(8, arange())
    # /opt/project/devkit/matlab/run_demoTracklets.m:89

    # 3D bounding box faces (indices for corners)
    face_idx = matlabarray(cat(1, 2, 6, 5, 2, 3, 7, 6, 3, 4, 8, 7, 4, 1, 5, 8))
    # /opt/project/devkit/matlab/run_demoTracklets.m:93

    # main loop (start at first image of sequence)
    img_idx = 0
    # /opt/project/devkit/matlab/run_demoTracklets.m:99
    while 1:

        # visualization update for next frame
        visualization('update', image_dir, gh, img_idx, nimages)
        for it in arange(1, numel(tracklets)).reshape(-1):
            # get relative tracklet frame index (starting at 0 with first appearance; 
            # xml data stores poses relative to the first frame where the tracklet appeared)
            pose_idx = img_idx - tracklets[it].first_frame + 1
            # /opt/project/devkit/matlab/run_demoTracklets.m:110
            # only draw tracklets that are visible in current frame
            if pose_idx < 1 or pose_idx > (size(tracklets[it].poses, 2)):
                continue
                # compute 3d object rotation in velodyne coordinates
                # VELODYNE COORDINATE SYSTEM:
                #   x -> facing forward
                #   y -> facing left
                #   z -> facing up
            R = matlabarray(cat([cos(rz[it](pose_idx)), - sin(rz[it](pose_idx)), 0],
                                [sin(rz[it](pose_idx)), cos(rz[it](pose_idx)), 0], [0, 0, 1]))
            # /opt/project/devkit/matlab/run_demoTracklets.m:122
            corners_3D = dot(R, cat([corners[it].x], [corners[it].y], [corners[it].z]))
            # /opt/project/devkit/matlab/run_demoTracklets.m:127
            corners_3D[1, :] = corners_3D[1, :] + t[it](1, pose_idx)
            # /opt/project/devkit/matlab/run_demoTracklets.m:128
            corners_3D[2, :] = corners_3D[2, :] + t[it](2, pose_idx)
            # /opt/project/devkit/matlab/run_demoTracklets.m:129
            corners_3D[3, :] = corners_3D[3, :] + t[it](3, pose_idx)
            # /opt/project/devkit/matlab/run_demoTracklets.m:130
            corners_3D = (dot(veloToCam[cam + 1], cat([corners_3D], [ones(1, size(corners_3D, 2))])))
            # /opt/project/devkit/matlab/run_demoTracklets.m:131
            orientation_3D = dot(R, cat([0.0, dot(0.7, l)], [0.0, 0.0], [0.0, 0.0]))
            # /opt/project/devkit/matlab/run_demoTracklets.m:134
            orientation_3D[1, :] = orientation_3D[1, :] + t[it](1, pose_idx)
            # /opt/project/devkit/matlab/run_demoTracklets.m:135
            orientation_3D[2, :] = orientation_3D[2, :] + t[it](2, pose_idx)
            # /opt/project/devkit/matlab/run_demoTracklets.m:136
            orientation_3D[3, :] = orientation_3D[3, :] + t[it](3, pose_idx)
            # /opt/project/devkit/matlab/run_demoTracklets.m:137
            orientation_3D = (dot(veloToCam[cam + 1], cat([orientation_3D], [ones(1, size(orientation_3D, 2))])))
            # /opt/project/devkit/matlab/run_demoTracklets.m:138
            if any(corners_3D[3, :] < 0.5) or any(orientation_3D[3, :] < 0.5):
                continue
            # project the 3D bounding box into the image plane
            corners_2D = projectToImage(corners_3D, K)
            # /opt/project/devkit/matlab/run_demoTracklets.m:146
            orientation_2D = projectToImage(orientation_3D, K)
            # /opt/project/devkit/matlab/run_demoTracklets.m:147
            drawBox3D(gh, occlusion[it](pose_idx), corners_2D, face_idx, orientation_2D)
            # compute and draw the 2D bounding box from the 3D box projection
            box.x1 = copy(min(corners_2D[1, :]))
            # /opt/project/devkit/matlab/run_demoTracklets.m:151
            box.x2 = copy(max(corners_2D[1, :]))
            # /opt/project/devkit/matlab/run_demoTracklets.m:152
            box.y1 = copy(min(corners_2D[2, :]))
            # /opt/project/devkit/matlab/run_demoTracklets.m:153
            box.y2 = copy(max(corners_2D[2, :]))
            # /opt/project/devkit/matlab/run_demoTracklets.m:154
            drawBox2D(gh, box, occlusion[it](pose_idx), tracklets[it].objectType)
        # force drawing and tiny user interface
        waitforbuttonpress
        key = get(gcf, 'CurrentCharacter')
        # /opt/project/devkit/matlab/run_demoTracklets.m:160
        if 'q' == lower(key):
            break
        else:
            if '-' == lower(key):
                img_idx = max(img_idx - 1, 0)
            # /opt/project/devkit/matlab/run_demoTracklets.m:163
            else:
                if 'x' == lower(key):
                    img_idx = min(img_idx + 100, nimages - 1)
                # /opt/project/devkit/matlab/run_demoTracklets.m:164
                else:
                    if 'y' == lower(key):
                        img_idx = max(img_idx - 100, 0)
                    # /opt/project/devkit/matlab/run_demoTracklets.m:165
                    else:
                        img_idx = min(img_idx + 1, nimages - 1)
                    # /opt/project/devkit/matlab/run_demoTracklets.m:166

    # clean up
    close_('all')
