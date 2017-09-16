import time
import os
from python.neural_network.Dataset import DataSet
from python.neural_network.GanNetworkVanilla import GanNetworkVanilla
from python.neural_network.GanNetworkSlim import GanNetworkSlim
import numpy as np
import pickle


def load_data(data_file):
    data = np.empty(shape=0)

    file = open(data_file, 'rb')
    drive_data = pickle.load(file)
    data = np.concatenate((data, drive_data))
    file.close()

    data_set = DataSet(data=data)
    return data_set


def network_train_test():
    data_file = 'tests/data/extracted/tracklets_photos_normalized.data'
    data_set = load_data(data_file)

    batch_size = 36
    z_dim = 100

    l1_ratio = 100
    epochs = 1
    gf_dim = 64  # (optional) Dimension of gen filters in first conv layer.
    df_dim = 64  # (optional) Dimension of discrim filters in first conv layer.
    gfc_dim = 1024  # (optional) Dimension of gen units for for fully connected layer.
    dfc_dim = 1024  # (optional) Dimension of discrim units for fully connected layer.
    c_dim = 3  # (optional) Dimension of image color. For grayscale input, set to 1, for colors, set to 3.
    learning_rate = 0.0002  # Learning rate of for adam
    beta1 = 0.5  # Momentum term of adam

    checkpoint_dir = os.path.join('tests', 'checkpoint')
    sample_dir = os.path.join('tests', 'samples')
    logs_dir = os.path.join('tests', 'logs')
    model_name = 'CGAN.model'

    network_slim = GanNetworkSlim(checkpoint_dir)
    image_size = data_set.get_image_size()
    y_dim = data_set.get_labels_dim()
    network_slim.build_model(image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1, df_dim, dfc_dim)
    network_slim.train(data_set, logs_dir, epochs, sample_dir, model_name)

    print("learning has ended")


def network_generate_test():
    data_file = 'tests/data/extracted/tracklets_photos_normalized.data'
    data_set = load_data(data_file)

    batch_size = 36
    z_dim = 100

    l1_ratio = 100
    gf_dim = 64  # (optional) Dimension of gen filters in first conv layer.
    df_dim = 64  # (optional) Dimension of discrim filters in first conv layer.
    gfc_dim = 1024  # (optional) Dimension of gen units for for fully connected layer.
    dfc_dim = 1024  # (optional) Dimension of discrim units for fully connected layer.
    c_dim = 3  # (optional) Dimension of image color. For grayscale input, set to 1, for colors, set to 3.
    learning_rate = 0.0002  # Learning rate of for adam
    beta1 = 0.5  # Momentum term of adam

    checkpoint_dir = os.path.join('tests', 'checkpoint')

    network = GanNetworkSlim(checkpoint_dir)
    image_size = data_set.get_image_size()
    y_dim = data_set.get_labels_dim()
    network.build_model(image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1, df_dim, dfc_dim)
    network.load()

    feature_vector = [0, 1, 2.8, 30 / 100, 1, 1]
    features = np.tile(feature_vector, [batch_size, 1])

    samples_dir = os.path.join('tests', 'samples_trained')
    suffix = 'testing'

    network.generate(features, samples_dir, suffix)

    print("learning has ended")


def main():
    network_train_test()
    network_generate_test()


if __name__ == '__main__':
    main()
