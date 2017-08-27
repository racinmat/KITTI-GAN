from python.neural_network.Dataset import DataSet
import tensorflow.contrib.slim as slim
from python.network_utils import *
import time
import tensorflow as tf
import numpy as np
import os
import pickle
from python.neural_network.train_gan import load_data
from tensorflow.python.framework.ops import GraphKeys


def main():
    data_dir = 'data/extracted'
    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]

    input_prefix = 'tracklets_photos_normalized_'
    resolution = (32, 32)

    data_set = load_data(resolution, drives, input_prefix, data_dir)

    # batch_size = 64
    batch_size = 36
    z_dim = 100

    l1_ratio = 100
    epochs = 5000
    gf_dim = 64  # (optional) Dimension of gen filters in first conv layer.
    df_dim = 64  # (optional) Dimension of discrim filters in first conv layer.
    gfc_dim = 1024  # (optional) Dimension of gen units for for fully connected layer.
    dfc_dim = 1024  # (optional) Dimension of discrim units for fully connected layer.
    c_dim = 3  # (optional) Dimension of image color. For grayscale input, set to 1, for colors, set to 3.
    learning_rate = 0.0002  # Learning rate of for adam
    beta1 = 0.5  # Momentum term of adam

    current_time = str(int(time.time()))
    sample_dir = os.path.join('samples', current_time)  # Directory name to save the image samples
    checkpoint_dir = os.path.join('checkpoint', current_time)  # Directory name to save the checkpoints
    logs_dir = os.path.join('logs', current_time)
    model_name = 'CGAN.model'

    d_optim, g_optim, summ, sampler, sess, d_loss_fake, d_loss_real, x, y, z, d_loss, g_loss, image_size \
        = build_gan(data_set, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1, df_dim,
                    dfc_dim)

    train_network(logs_dir, epochs, batch_size, z_dim, sess, d_optim, g_optim, d_loss_fake,
                  d_loss_real, x, y, z, data_set, d_loss, g_loss, summ, sampler, sample_dir, checkpoint_dir,
                  image_size, model_name)

    print("learning has ended")


if __name__ == '__main__':
    main()