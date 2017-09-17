import math

from devkit.python.utils import transform_to_range
from python.network_utils import *
from python.neural_network.GanNetworkSlim import GanNetworkSlim


def main():
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
    network.load()

    feature_vector = [0, 1, 2.8, 30 / 100, 1, 1]
    features = np.tile(feature_vector, [batch_size, 1])

    samples_dir = os.path.join('tests', 'samples_trained')
    suffix = 'testing'

    network.generate(features, samples_dir, suffix)

    print("images saved to dir {}".format(samples_dir))


if __name__ == '__main__':
    main()
