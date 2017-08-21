import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    show_data_image = True
    show_metadata_image = True

    data_dir = 'data/extracted'
    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]

    input_prefix = 'tracklets_points_normalized_'
    input_suffix = ''
    # input_suffix = '_0_20'
    resolution = '32_32'
    # resolution = '64_64'

    directory = 'images/' + resolution
    directory = directory + '/normalized/'

    cam = 2
    drive_dir = './data/2011_09_26/2011_09_26_'
    calib_dir = './data/2011_09_26'

    if not os.path.exists(directory):
        os.makedirs(directory)

    features = np.empty(shape=0)

    for i, drive in enumerate(drives):
        filename = data_dir + '/' + input_prefix + drive + '_' + resolution + input_suffix + '.data'
        file = open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        current_dir = drive_dir + drive
        samples = len(data)
        print("processing: {:s} with {:d} samples".format(filename, samples))
        for j, sample in enumerate(data):
            feature_vector = np.array(sample['x'])
            features = np.vstack((features, feature_vector)) if features.size else feature_vector

    nbins = 500
    fig = plt.figure()
    for i in range(features.shape[1]):
        fig.add_subplot(features.shape[1], 1, i+1)
        plt.hist(x=features[:, i], bins=nbins)
        plt.title('hist of {}'.format(i))

    plt.tight_layout()
    plt.show()