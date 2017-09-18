import unittest
import os
from python.neural_network.Dataset import DataSet
from python.neural_network.GanNetworkSlim import GanNetworkSlim
import numpy as np
import pickle

from python.neural_network.GanNetworkSlimDropouts import GanNetworkSlimDropouts
from python.neural_network.GanNetworkSlimLabelSmoothing import GanNetworkSlimLabelSmoothing


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

        network = GanNetworkSlim(checkpoint_dir=checkpoint_dir, name='gan_slim')
        image_size = data_set.get_image_size()
        y_dim = data_set.get_labels_dim()
        network.build_model(image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate,
                            beta1, df_dim, dfc_dim)
        network.train(data_set, logs_dir, epochs, sample_dir, train_test_ratios=[0.5, 0.5])
        self.assert_files_created('gan_slim')

        network = GanNetworkSlimLabelSmoothing(checkpoint_dir=checkpoint_dir, name='gan_slim_label_smoothing')
        image_size = data_set.get_image_size()
        y_dim = data_set.get_labels_dim()
        network.build_model(image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate,
                            beta1, df_dim, dfc_dim, smooth=0.1)
        network.train(data_set, logs_dir, epochs, sample_dir, train_test_ratios=[0.5, 0.5])
        self.assert_files_created('gan_slim_label_smoothing')

        network = GanNetworkSlimDropouts(checkpoint_dir=checkpoint_dir, name='gan_slim_dropouts')
        image_size = data_set.get_image_size()
        y_dim = data_set.get_labels_dim()
        network.build_model(image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate,
                            beta1, df_dim, dfc_dim, dropout_rate=0.5)
        network.train(data_set, logs_dir, epochs, sample_dir, train_test_ratios=[0.5, 0.5])
        self.assert_files_created('gan_slim_dropouts')

    def assert_files_created(self, network_name):
        self.assertTrue(os.path.exists(os.path.join('tests', 'checkpoint', 'KITTI_36_32_32', network_name + '-0.index')))
        self.assertTrue(os.path.exists(os.path.join('tests', 'checkpoint', 'KITTI_36_32_32', network_name + '-0.meta')))
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
