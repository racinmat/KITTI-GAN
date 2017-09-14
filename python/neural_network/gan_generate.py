import math

from devkit.python.utils import transform_to_range
from python.network_utils import *


def main():
    network_name = '1503666003/KITTI_36_32_32'
    checkpoint_dir = os.path.join('./checkpoint', network_name)
    samples_dir = os.path.join('./samples_trained', network_name)
    suffix = 'fixed_features_fabricated'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    load(sess, checkpoint_dir)

    graph = tf.get_default_graph()
    z = graph.get_tensor_by_name('z:0')
    y = graph.get_tensor_by_name('y:0')

    # samples
    samples_num = 1
    for idx in range(samples_num):
        batch_size = int(z.shape[0])
        z_dim = int(z.shape[1])
        samples_in_batch = batch_size
        z_sample = sample_z(samples_in_batch, z_dim)
        # feature_vector = [0, 1, 30/100, 1, 1]
        feature_vector = [0, 1, 2.8, 30/100, 1, 1]
        # feature_vector = [-0.11912173, 1.00015249, 2.87348039, 0.43911632, 1., 1.]
        # feature_vector = [0.5, 0.1, 30/100, 0.5, 1]
        # feature_vector = [1, 0.5, 0.1, 30/100, 0.5, 1]
        features = np.tile(feature_vector, [batch_size, 1])

        # features = []
        # for i in range(batch_size):
        #     if idx is 1:
        #         feature_vector = [1, transform_to_range(0, 36, 0, 1, i), 1, 30 / 100, 1, 1]
        #     else:
        #         feature_vector = [1, 1, transform_to_range(0, 36, 0, 1, i), 30 / 100, 1, 1]
        #     features.append(feature_vector)

        image_frame_dim = int(math.ceil(batch_size ** .5))

        sampler = graph.get_tensor_by_name('generator/generator:0')    # this is last layer of generator layer
        samples = sess.run(sampler, feed_dict={z: z_sample, y: features})

        if not os.path.exists(os.path.dirname(samples_dir)):
            os.makedirs(os.path.dirname(samples_dir))

        save_images(samples, [image_frame_dim, image_frame_dim], '{}/test_{}_{:d}.png'.format(samples_dir, suffix, idx))
        print("images saved to dir {}".format(samples_dir))

if __name__ == '__main__':
    main()
