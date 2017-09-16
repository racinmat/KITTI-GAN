import os
import numpy as np
import scipy.misc
import tensorflow as tf


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    # this propagates labels to other layers of neural network,
    # probably according to https://arxiv.org/pdf/1611.01455.pdf
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x, name=name)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


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


def model_dir(batch_size, image_size):
    return "{}_{}_{}_{}".format(
        'KITTI', batch_size,
        image_size[1], image_size[0])


def save(checkpoint_dir, step, batch_size, image_size, saver, sess, model_name):
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir(batch_size, image_size))

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,
               os.path.join(checkpoint_dir, model_name),
               global_step=step)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def inverse_transform(images):
    """
    :type images: np.ndarray
    :rtype np.ndarray
    """
    return (images + 1.) / 2.


def save_images(images, size, image_path):
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))

    return imsave(inverse_transform(images), size, image_path)


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def load(session, checkpoint_dir):
    import re
    print(" [*] Loading last checkpoint")

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
        data_file = os.path.join(checkpoint_dir, checkpoint_name)
        meta_file = data_file + '.meta'
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(session, data_file)
        counter = int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
        print(" [*] Success to read {}".format(checkpoint_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0

