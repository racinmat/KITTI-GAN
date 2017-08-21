import math
from python.network_utils import *


def main():
    checkpoint_dir = './checkpoint/KITTI_64_32_32/'
    samples_dir = './samples_trained'
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
        feature_vector = [1, 1, 30, 32, 32]
        features = np.tile(feature_vector, [batch_size, 1])

        image_frame_dim = int(math.ceil(batch_size ** .5))

        sampler = graph.get_tensor_by_name('generator/generator:0')    # this is last layer of generator layer
        samples = sess.run(sampler, feed_dict={z: z_sample, y: features})

        save_images(samples, [image_frame_dim, image_frame_dim], '{}/test_{:d}.png'.format(samples_dir, idx))

if __name__ == '__main__':
    main()
