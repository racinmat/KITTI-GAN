import time
import os

from python.neural_network.GanNetworkSlimLabelSmoothing import GanNetworkSlimLabelSmoothing
from python.neural_network.GanNetworkVanilla import GanNetworkVanilla
from python.neural_network.GanNetworkSlim import GanNetworkSlim
from python.neural_network.train_gan import load_data
import tensorflow as tf


def main():
    data_dir = 'data/extracted'
    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]

    input_prefix = 'tracklets_photos_normalized_'
    # resolution = (32, 32)
    resolution = (64, 64)

    data_set = load_data(resolution, drives, input_prefix, data_dir)

    # batch_size = 64
    batch_size = 36
    z_dim = 100

    l1_ratio = 100
    epochs = 400
    gf_dim = 64  # (optional) Dimension of gen filters in first conv layer.
    df_dim = 64  # (optional) Dimension of discrim filters in first conv layer.
    gfc_dim = 1024  # (optional) Dimension of gen units for for fully connected layer.
    dfc_dim = 1024  # (optional) Dimension of discrim units for fully connected layer.
    c_dim = 3  # (optional) Dimension of image color. For grayscale input, set to 1, for colors, set to 3.
    learning_rate = 0.0002  # Learning rate of for adam
    beta1 = 0.5  # Momentum term of adam
    smooth = 0.1  # used for one sided label smoothing

    # GPU settings
    gpu = 1  # use the second GPU
    available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = available_devices[gpu]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    current_time = str(int(time.time()))
    sample_dir = os.path.join('samples', current_time)  # Directory name to save the image samples
    checkpoint_dir = os.path.join('checkpoint', current_time)  # Directory name to save the checkpoints
    logs_dir = os.path.join('logs', current_time)

    network_slim = GanNetworkSlimLabelSmoothing(checkpoint_dir, config=config)
    image_size = data_set.get_image_size()
    y_dim = data_set.get_labels_dim()
    network_slim.build_model(image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1, df_dim, dfc_dim, smooth)
    network_slim.train(data_set, logs_dir, epochs, sample_dir)

    print("learning has ended")


if __name__ == '__main__':
    main()
