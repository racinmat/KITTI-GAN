import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from python.data_utils import load_features

if __name__ == '__main__':
    show_data_image = True
    show_metadata_image = True

    data_dir = 'data/extracted'

    # input_prefix = 'tracklets_points_normalized_'
    input_prefix = 'tracklets_photos_normalized_'
    input_suffix = ''
    # input_suffix = '_0_20'
    # resolution = '32_32'
    resolution = '64_64'

    # directory = os.path.join('images', resolution, 'normalized')
    #
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    features = load_features(input_prefix, resolution, input_suffix)

    nbins = 500
    fig = plt.figure()
    for i in range(features.shape[1]):
        fig.add_subplot(features.shape[1], 1, i+1)
        plt.hist(x=features[:, i], bins=nbins)
        plt.title('hist of {}'.format(i))

    plt.tight_layout()
    plt.show()