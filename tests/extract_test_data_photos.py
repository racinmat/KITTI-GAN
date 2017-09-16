import matplotlib
# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

from python.data_preparation.DatasetFilterer import DatasetFilterer
from python.data_preparation.extract_data import get_x_y_data_for, load_tracklets
import glob
from PIL import Image
import pickle
from python.data_utils import is_tracklet_seen
from devkit.python.utils import timeit
import numpy as np
import os


@timeit
def extract_data(output_file):
    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]
    drive_dir = './data/2011_09_26/2011_09_26_'
    calib_dir = './data/2011_09_26'

    limit = 36

    cam = 2

    min_distance = 15
    max_distance = 45
    treshold_degrees = None
    filterer = DatasetFilterer(min_distance=min_distance, max_distance=max_distance, treshold_degrees=treshold_degrees)
    samples = 0
    data = []

    for i, drive in enumerate(drives):
        if samples > limit:
            break

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

        tracklets = load_tracklets(base_dir=current_dir)

        for frame in range(start, end):
            if samples > limit:
                break

            for j, tracklet in enumerate(tracklets):
                if samples > limit:
                    break

                if not is_tracklet_seen(tracklet=tracklet, frame=frame, calib_dir=calib_dir, cam=cam):
                    continue

                if not filterer.is_for_dataset(tracklet=tracklet, frame=frame, cam=cam, calib_dir=calib_dir):
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
                samples += 1

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        file = open(output_file, 'wb')
        pickle.dump(data, file)
        print('data saved to file: {}, extracted {} samples.'.format(output_file, len(data)))
        file.close()


def normalize_data(input_file, output_file, new_size):
    print("processing: " + input_file)
    file = open(input_file, 'rb')
    data = pickle.load(file)
    file.close()
    for i, pair in enumerate(data):
        img = Image.fromarray(pair['y'])
        ratio = min(new_size[0] / img.size[0], new_size[1] / img.size[1])

        # resize image
        # img.save('temp_orig.png')
        new_img = img.resize((round(img.size[0] * ratio), round(img.size[1] * ratio)))
        # new_img.save('temp_resized.png')

        # fill missing places with white
        white = {'L': 255, 'RGB': (255, 255, 255), 'RGBA': (255, 255, 255, 0)}
        black = {'L': 0, 'RGB': (0, 0, 0), 'RGBA': (0, 0, 0, 0)}
        bg = Image.new(mode=img.mode, size=new_size, color=black[img.mode])
        bg.paste(new_img, (0, 0, new_img.size[0], new_img.size[1]))  # Not centered, top-left corner
        # bg.save('temp_resized_padded.png')

        pair['y'] = np.array(bg)
        pair['x'][3] = pair['x'][3] / 100  # cars are distant up to 100 meters, so this is normalizing the distance
        pair['x'].append(new_img.size[0] / new_size[
            0])  # because of regularization, I want to keep size as ratio, in values from 0 to 1
        pair['x'].append(new_img.size[1] / new_size[1])

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    file = open(output_file, 'wb')
    pickle.dump(data, file)
    print("data written to file: " + output_file)
    file.close()


if __name__ == '__main__':
    extraction_output_file = 'tests/data/extracted/tracklets_photos.data'
    normalized_output_file = 'tests/data/extracted/tracklets_photos_normalized.data'
    resolution = (32, 32)
    extract_data(extraction_output_file)
    normalize_data(extraction_output_file, normalized_output_file, resolution)

    print("extraction done")
