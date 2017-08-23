import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
from python.data_preparation.extract_data import is_for_dataset, get_x_y_data_for, load_tracklets

matplotlib.use('Agg')

import glob
from PIL import Image
import pickle
from python.data_utils import is_tracklet_seen
from devkit.python.utils import timeit


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

            # if not velodyne_data_exist(current_dir, frame):
            #     continue

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
                                          with_image=True,
                                          with_velo=False,
                                          grayscale=False,
                                          )

                # visualization of sample
                # buf, im = sample_to_image(sample, cam, calib_dir, current_dir)
                # im.save('images/extraction/' + drive + '_{:d}_src_frame_{:d}.png'.format(j, frame))
                # buf.close()
                # end of visualization

                data.append(sample)

        file_name = 'data/extracted/tracklets_photos_' + drive
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
