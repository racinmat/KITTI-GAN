from python.network_utils import *
from python.neural_network.GanNetworkSlim import GanNetworkSlim
import tensorflow as tf


def main():
    batch_size = 36
    z_dim = 100
    image_size = (64, 64)
    sampler_name = 'generator/g_h3/Sigmoid:0'

    # base_name = '.'
    base_name = '/datagrid/personal/racinmat'
    checkpoints_dir = os.path.join(base_name, 'checkpoint')

    network_names = [
        # '1505772010',
        # '1505772022',
        '1505828231',
        '1505828317',
        '1505950038',
        '1506127735',
        '1506127768',
    ]

    feature_vector = [0, 1, 2.8, 30 / 100, 1, 1]
    features = np.tile(feature_vector, [batch_size, 1])

    samples_dir = os.path.join('samples_trained')

    tf.logging.set_verbosity(tf.logging.DEBUG)

    for name in network_names:
        suffix = name
        network_checkpoint_dir = os.path.join(checkpoints_dir, name)
        network = GanNetworkSlim(network_checkpoint_dir)
        network.build_empty_model(image_size, batch_size, z_dim)
        network.load_with_structure(sampler_name)

        network.generate(features, samples_dir, suffix)
        tf.logging.info("generated samples for 1 network to: {}".format(samples_dir))

    tf.logging.info("images saved to dir {}".format(samples_dir))


if __name__ == '__main__':
    main()
