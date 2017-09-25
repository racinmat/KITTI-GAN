import time
import os
import pprint
from python.neural_network.GanNetworkSlimDropouts import GanNetworkSlimDropouts
from python.neural_network.GanNetworkSlimLabelSmoothing import GanNetworkSlimLabelSmoothing
from python.neural_network.GanNetworkSlim import GanNetworkSlim
from python.neural_network.train_gan import load_data
import tensorflow as tf
import sys
import logging
import sys

flags = tf.app.flags
flags.DEFINE_integer("epoch", 400, "Epoch to train [400]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 36, "The size of batch images [36]")
flags.DEFINE_integer("l1_ratio", 100, "Ratio between GAN and L1 loss [100]")
flags.DEFINE_integer("gpu", 1, "GPU number (indexed from 0 [1], used when multiple GPUs are used.")
flags.DEFINE_string("type", 'basic', "Type of network [basic, label_smoothing, dropouts]")
flags.DEFINE_string("output_dir", '.', "Location of output dir")
flags.DEFINE_integer("dropout_rate", 0.5, "Dropout rate [0.5]")
flags.DEFINE_integer("smooth", 0.1, "Smoothing [0.1]")
flags.DEFINE_string("z_sampling", 'normal', "Sampling type (uniform|normal) [normal]")
FLAGS = flags.FLAGS


def main(__):
    current_time = str(int(time.time()))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    output_dir = FLAGS.output_dir
    sample_dir = os.path.join(output_dir, 'samples', current_time)  # Directory name to save the image samples
    checkpoint_dir = os.path.join(output_dir, 'checkpoint', current_time)  # Directory name to save the checkpoints
    logs_dir = os.path.join(output_dir, 'logs', current_time)
    screen_output_dir = os.path.join(output_dir, 'output', current_time)

    if not os.path.exists(screen_output_dir):
        os.makedirs(screen_output_dir)

    ch = logging.FileHandler(screen_output_dir+'/out.log', mode='w')
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

    arguments = pprint.PrettyPrinter().pformat(flags.FLAGS.__flags)

    logging.getLogger().info('arguments:')
    logging.getLogger().info(arguments)

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
    batch_size = FLAGS.batch_size
    z_dim = 100

    l1_ratio = FLAGS.l1_ratio
    epochs = FLAGS.epoch
    gf_dim = 64  # (optional) Dimension of gen filters in first conv layer.
    df_dim = 64  # (optional) Dimension of discrim filters in first conv layer.
    gfc_dim = 1024  # (optional) Dimension of gen units for for fully connected layer.
    dfc_dim = 1024  # (optional) Dimension of discrim units for fully connected layer.
    c_dim = 3  # (optional) Dimension of image color. For grayscale input, set to 1, for colors, set to 3.
    learning_rate = FLAGS.learning_rate  # Learning rate of for adam
    beta1 = FLAGS.beta1
    dropout_rate = FLAGS.dropout_rate
    smooth = FLAGS.smooth
    z_sampling = FLAGS.z_sampling

    # GPU settings
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    devices_environ_var = 'CUDA_VISIBLE_DEVICES'
    if devices_environ_var in os.environ:
        available_devices = os.environ[devices_environ_var].split(',')
        if len(available_devices):
            gpu = FLAGS.gpu  # use the second GPU
            os.environ[devices_environ_var] = available_devices[gpu]

    image_size = data_set.get_image_size()
    y_dim = data_set.get_labels_dim()

    if FLAGS.type == 'basic':
        network = GanNetworkSlim(checkpoint_dir, config=config)
        network.build_model(image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate,
                            beta1, df_dim, dfc_dim, z_sampling)
    elif FLAGS.type == 'dropouts':
        network = GanNetworkSlimDropouts(checkpoint_dir, config=config)
        network.build_model(image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate,
                            beta1, df_dim, dfc_dim, z_sampling, dropout_rate=dropout_rate)
    elif FLAGS.type == 'label_smoothing':
        network = GanNetworkSlimLabelSmoothing(checkpoint_dir, config=config)
        network.build_model(image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate,
                            beta1, df_dim, dfc_dim, z_sampling, smooth)
    else:
        raise Exception("Wrong network type")

    network.train(data_set, logs_dir, epochs, sample_dir, train_test_ratios=[0.8, 0.2])

    logging.getLogger().info("learning has ended")


if __name__ == '__main__':
    tf.app.run()
