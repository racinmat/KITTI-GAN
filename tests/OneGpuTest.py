import unittest
import os
from python.neural_network.Dataset import DataSet
from python.neural_network.GanNetworkSlim import GanNetworkSlim
import numpy as np
import pickle

from python.neural_network.GanNetworkSlimDropouts import GanNetworkSlimDropouts
from python.neural_network.GanNetworkSlimLabelSmoothing import GanNetworkSlimLabelSmoothing
import tensorflow as tf


class GanTest(unittest.TestCase):
    def load_data(self, data_file):
        data = np.empty(shape=0)

        file = open(data_file, 'rb')
        drive_data = pickle.load(file)
        data = np.concatenate((data, drive_data))
        file.close()

        data_set = DataSet(data=data)
        return data_set

    def _test_network_train(self):
        data_file = 'tests/data/extracted/tracklets_photos_normalized.data'
        data_set = self.load_data(data_file)

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

        gpu = 1 # use the second GPU
        available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES'] = available_devices[gpu]

        # config = tf.ConfigProto(log_device_placement=True)
        config = None

        network = GanNetworkSlim(checkpoint_dir=checkpoint_dir, name='gan_slim', config=config)
        image_size = data_set.get_image_size()
        y_dim = data_set.get_labels_dim()
        network.build_model(image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate,
                            beta1, df_dim, dfc_dim)
        network.train(data_set, logs_dir, epochs, sample_dir)
        self.assert_files_created('gan_slim')

    def assert_files_created(self, network_name):
        self.assertTrue(os.path.exists(os.path.join('tests', 'checkpoint', 'KITTI_36_32_32', network_name + '-1.index')))
        self.assertTrue(os.path.exists(os.path.join('tests', 'checkpoint', 'KITTI_36_32_32', network_name + '-1.meta')))
        self.assertTrue(os.path.exists(os.path.join('tests', 'checkpoint', 'KITTI_36_32_32', 'checkpoint')))
        self.assertTrue(os.path.exists(os.path.join('tests', 'logs', network_name)))
        self.assertTrue(os.path.exists(os.path.join('tests', 'samples', 'train_00_0000.png')))
        self.assertFalse(os.path.exists(os.path.join('tests', 'samples_trained', 'test_testing_0.png')))

    def _test_network_generate(self):
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
        image_size = (32, 32)
        y_dim = 6
        network.build_model(image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate,
                            beta1, df_dim, dfc_dim)
        loaded, counter = network.load()

        feature_vector = [0, 1, 2.8, 30 / 100, 1, 1]
        features = np.tile(feature_vector, [batch_size, 1])

        samples_dir = os.path.join('tests', 'samples_trained')
        suffix = 'testing'

        network.generate(features, samples_dir, suffix)

        self.assertTrue(loaded)
        self.assertTrue(os.path.exists(os.path.join('tests', 'samples_trained', 'test_testing_0.png')))

    def test_network(self):
        self._test_network_train()
        self._test_network_generate()

    def tearDown(self):
        import shutil
        shutil.rmtree(os.path.join('tests', 'samples'))
        shutil.rmtree(os.path.join('tests', 'checkpoint'))
        shutil.rmtree(os.path.join('tests', 'logs'))
        shutil.rmtree(os.path.join('tests', 'samples_trained'))


if __name__ == '__main__':
    unittest.main()
