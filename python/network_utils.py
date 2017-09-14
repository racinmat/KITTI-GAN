import os
import numpy as np
import scipy.misc
import tensorflow as tf
import time
from tensorflow.python.framework.ops import GraphKeys
import tensorflow.contrib.slim as slim

from python.neural_network.GeneratorFactory import GeneratorFactory


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    # this propagates labels to other layers of neural network, probably according to https://arxiv.org/pdf/1611.01455.pdf
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="BatchNorm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", biases)

        return conv


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x, name=name)


def linear(input_, output_size, scope=None, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        w = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", bias)

        if with_w:
            return tf.matmul(input_, w) + bias, w, bias
        else:
            return tf.matmul(input_, w) + bias


def discriminator(x, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        d_bn1 = BatchNorm(name='d_bn1')
        d_bn2 = BatchNorm(name='d_bn2')

        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
        x = conv_cond_concat(x, yb)

        h0 = lrelu(conv2d(x, c_dim + y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(d_bn1(conv2d(h0, df_dim + y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [batch_size, -1])
        h1 = tf.concat([h1, y], 1)

        h2 = lrelu(d_bn2(linear(h1, dfc_dim, 'd_h2_lin')))
        h2 = tf.concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')

        return tf.nn.sigmoid(h3), h3


def generator(z, y, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim):
    with tf.variable_scope("generator"):
        g_bn0 = BatchNorm(name='g_bn0')
        g_bn1 = BatchNorm(name='g_bn1')
        g_bn2 = BatchNorm(name='g_bn2')

        # taken from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
        s_h, s_w = image_size[1], image_size[0]
        s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
        s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
        z = tf.concat([z, y], 1)

        h0 = tf.nn.relu(
            g_bn0(linear(z, gfc_dim, 'g_h0_lin')))
        h0 = tf.concat([h0, y], 1)

        h1 = tf.nn.relu(g_bn1(
            linear(h0, gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(g_bn2(deconv2d(h1, [batch_size, s_h2, s_w2, gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        h3 = tf.nn.sigmoid(deconv2d(h2, [batch_size, s_h, s_w, c_dim], name='g_h3'))

        return tf.identity(h3, 'generator')


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


def build_gan(data_set, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1, df_dim, dfc_dim):
    image_size = data_set.get_image_size()
    y_dim = data_set.get_labels_dim()

    x = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], c_dim], name='x')
    y = tf.placeholder(tf.float32, shape=[batch_size, y_dim], name='y')
    z = tf.placeholder(tf.float32, shape=[batch_size, z_dim], name='z')
    G_factory = GeneratorFactory(z, y, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim)
    G = G_factory.create(is_training=True, reuse=False)
    # G = generator(z, y, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim)
    D_real, D_logits_real = discriminator(x, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=False)
    sampler = G
    D_fake, D_logits_fake = discriminator(G, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=True)

    tf.summary.histogram("z", z)
    tf.summary.histogram("d_real", D_real)
    tf.summary.histogram("d_fake", D_fake)
    tf.summary.image("g", G)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.zeros_like(D_fake)))
    # g_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.ones_like(D_fake)))

    # For generator we use traditional GAN objective as well as L1 loss
    # L1 added from https://github.com/awjuliani/Pix2Pix-Film/blob/master/Pix2Pix.ipynb
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.ones_like(D_fake))) + \
               l1_ratio * tf.reduce_mean(tf.abs(G - x))  # This optimizes the generator.

    tf.summary.scalar("d_loss_real", d_loss_real)
    tf.summary.scalar("d_loss_fake", d_loss_fake)

    d_loss = d_loss_real + d_loss_fake

    tf.summary.scalar("d_loss", d_loss)
    tf.summary.scalar("g_loss", g_loss)

    d_vars = slim.get_variables(scope='discriminator', collection=GraphKeys.TRAINABLE_VARIABLES)
    g_vars = slim.get_variables(scope='generator', collection=GraphKeys.TRAINABLE_VARIABLES)

    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    summ = tf.summary.merge_all()
    return d_optim, g_optim, summ, sampler, sess, d_loss_fake, d_loss_real, x, y, z, d_loss, g_loss, image_size


def build_gan_slim(data_set, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1, df_dim, dfc_dim):
    image_size = data_set.get_image_size()
    y_dim = data_set.get_labels_dim()

    x = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], c_dim], name='x')
    y = tf.placeholder(tf.float32, shape=[batch_size, y_dim], name='y')
    z = tf.placeholder(tf.float32, shape=[batch_size, z_dim], name='z')
    G_factory = GeneratorFactory(z, y, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim)
    G_2 = G_factory.create()
    G = generator(z, y, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim)
    D_real, D_logits_real = discriminator(x, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=False)
    sampler = G
    D_fake, D_logits_fake = discriminator(G, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=True)

    tf.summary.histogram("z", z)
    tf.summary.histogram("d_real", D_real)
    tf.summary.histogram("d_fake", D_fake)
    tf.summary.image("g", G)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.zeros_like(D_fake)))
    # g_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.ones_like(D_fake)))

    # For generator we use traditional GAN objective as well as L1 loss
    # L1 added from https://github.com/awjuliani/Pix2Pix-Film/blob/master/Pix2Pix.ipynb
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.ones_like(D_fake))) + \
               l1_ratio * tf.reduce_mean(tf.abs(G - x))  # This optimizes the generator.

    tf.summary.scalar("d_loss_real", d_loss_real)
    tf.summary.scalar("d_loss_fake", d_loss_fake)

    d_loss = d_loss_real + d_loss_fake

    tf.summary.scalar("d_loss", d_loss)
    tf.summary.scalar("g_loss", g_loss)

    d_vars = slim.get_variables(scope='discriminator', collection=GraphKeys.TRAINABLE_VARIABLES)
    g_vars = slim.get_variables(scope='generator', collection=GraphKeys.TRAINABLE_VARIABLES)

    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    summ = tf.summary.merge_all()
    return d_optim, g_optim, summ, sampler, sess, d_loss_fake, d_loss_real, x, y, z, d_loss, g_loss, image_size

def train_network(logs_dir, epochs, batch_size, z_dim, sess, d_optim, g_optim, d_loss_fake,
                  d_loss_real, x, y, z, data_set, d_loss, g_loss, summ, sampler, sample_dir, checkpoint_dir,
                  image_size, model_name):

    if not os.path.exists(os.path.dirname(logs_dir)):
        os.makedirs(os.path.dirname(logs_dir))

    writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

    saver = tf.train.Saver()

    counter = 0
    start_time = time.time()

    print("Starting to learn for {} epochs.".format(epochs))
    for epoch in range(epochs):
        num_batches = int(data_set.num_batches(batch_size))
        for i in range(num_batches):
            x_batch, y_batch = data_set.next_batch(batch_size)
            z_batch = sample_z(batch_size, z_dim)

            # Update D network
            _, errD_fake, errD_real = sess.run([d_optim, d_loss_fake, d_loss_real], feed_dict={
                x: x_batch,
                y: y_batch,
                z: z_batch,
            })

            # Update G network
            _ = sess.run([g_optim], feed_dict={
                x: x_batch,
                y: y_batch,
                z: z_batch,
            })

            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            _, errG = sess.run([g_optim, g_loss], feed_dict={
                x: x_batch,
                y: y_batch,
                z: z_batch,
            })

            # run summary of all
            summary_str = sess.run(summ, feed_dict={x: x_batch, z: z_batch, y: y_batch})
            writer.add_summary(summary_str, counter)

            counter += 1
            summary_string = "Epoch: {:2d} {:2d}/{:2d} counter: {:3d} time: {:4.4f}, d_loss: {:.6f}, g_loss: {:.6f}"
            print(summary_string.format(epoch, i, num_batches, counter, time.time() - start_time, errD_fake + errD_real,
                                        errG))

            if np.mod(counter, 100) == 1:
                try:
                    samples = sess.run(sampler, feed_dict={
                        z: z_batch,
                        y: y_batch
                    })
                    d_loss_val, g_loss_val = sess.run([d_loss, g_loss], feed_dict={
                        z: z_batch,
                        x: x_batch,
                        y: y_batch
                    })
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, i))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss_val, g_loss_val))
                except Exception as e:
                    print("pic saving error:")
                    print(e)
                    raise e

            if np.mod(counter, 800) == 2:
                save(checkpoint_dir, counter, batch_size, image_size, saver, sess, model_name)
                print("saved after {}. iteration".format(counter))

    writer.flush()
    writer.close()
    save(checkpoint_dir, counter, batch_size, image_size, saver, sess, model_name)
