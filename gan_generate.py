import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import os
import scipy.misc
import math


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def model_dir(batch_size, image_size):
    return "{}_{}_{}_{}".format(
        'KITTI', batch_size,
        image_size[1], image_size[0])


def save(checkpoint_dir, step, batch_size, image_size, saver, sess):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir(batch_size, image_size))

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,
               os.path.join(checkpoint_dir, model_name),
               global_step=step)


def load(session, checkpoint_dir):
    import re
    print(" [*] Loading last checkpoint")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        data_file = os.path.join(checkpoint_dir, ckpt_name)
        meta_file = data_file + '.meta'
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(session, data_file)
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def inverse_transform(images):
    return (images + 1.) / 2.


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def main():
    checkpoint_dir = './checkpoint/KITTI_64_32_32/'
    samples_dir = './samples_trained'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    load(sess, checkpoint_dir)

    graph = tf.get_default_graph()
    Z = graph.get_tensor_by_name('Z:0')
    y = graph.get_tensor_by_name('y:0')

    # samples
    samples_num = 1
    for idx in range(samples_num):
        batch_size = int(Z.shape[0])
        Z_dim = int(Z.shape[1])
        samples_in_batch = batch_size
        Z_sample = sample_Z(samples_in_batch, Z_dim)
        feature_vector = [1, 1, 30, 32, 32]
        features = np.tile(feature_vector, [batch_size, 1])

        image_frame_dim = int(math.ceil(batch_size ** .5))

        sampler = graph.get_tensor_by_name('generator/generator:0')    # this is last layer of generator layer
        samples = sess.run(sampler, feed_dict={Z: Z_sample, y: features})

        save_images(samples, [image_frame_dim, image_frame_dim], '{}/test_{:d}.png'.format(samples_dir, idx))

if __name__ == '__main__':
    main()
