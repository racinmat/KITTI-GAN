import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

import time
import tensorflow as tf
import numpy as np
import os
import pickle
from Dataset import DataSet
import scipy.misc


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


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
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


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def discriminator(x, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')

        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
        x = conv_cond_concat(x, yb)

        h0 = lrelu(conv2d(x, c_dim + y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(d_bn1(conv2d(h0, df_dim + y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [batch_size, -1])
        h1 = tf.concat([h1, y], 1)

        h2 = lrelu(d_bn2(linear(h1, dfc_dim, 'd_h2_lin')))
        # h2 = lrelu(d_bn2(linear(h0, dfc_dim, 'd_h2_lin')))
        h2 = tf.concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')

        return tf.nn.sigmoid(h3), h3


def generator(z, y, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim):
    with tf.variable_scope("generator") as scope:
        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')

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


def sample_Z(m, n):
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


def save(checkpoint_dir, step, batch_size, image_size, saver, sess):
    model_name = "DCGAN.model"
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


def main():
    data_dir = 'data/extracted'
    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]

    input_prefix = 'tracklets_points_normalized_'
    resolution = (32, 32)
    resolution_string = '{:d}_{:d}'.format(resolution[0], resolution[1])

    data = np.empty(shape=0)

    for i, drive in enumerate(drives):
        filename = data_dir + '/' + input_prefix + drive + '_' + resolution_string + '.data'
        file = open(filename, 'rb')
        drive_data = pickle.load(file)
        data = np.concatenate((data, drive_data))
        file.close()

    dataset = DataSet(data=data)

    batch_size = 64
    Z_dim = 100
    image_size = dataset.get_image_size()
    y_dim = dataset.get_labels_dim()

    epochs = 500
    gf_dim = 64  # (optional) Dimension of gen filters in first conv layer.
    df_dim = 64  # (optional) Dimension of discrim filters in first conv layer.
    gfc_dim = 1024  # (optional) Dimension of gen units for for fully connected layer.
    dfc_dim = 1024  # (optional) Dimension of discrim units for fully connected layer.
    c_dim = 1  # (optional) Dimension of image color. For grayscale input, set to 1, for colors, set to 3.
    learning_rate = 0.0002  # Learning rate of for adam
    beta1 = 0.5  # Momentum term of adam
    sample_dir = os.path.join('samples', str(int(time.time())))  # Directory name to save the image samples
    checkpoint_dir = os.path.join('checkpoint', str(int(time.time())))  # Directory name to save the checkpoints

    X = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], c_dim], name='X')
    y = tf.placeholder(tf.float32, shape=[batch_size, y_dim], name='y')
    Z = tf.placeholder(tf.float32, shape=[batch_size, Z_dim], name='Z')
    G = generator(Z, y, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim)
    D_real, D_logits_real = discriminator(X, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=False)
    sampler = G
    D_fake, D_logits_fake = discriminator(G, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=True)

    z_sum = tf.summary.histogram("z", Z)
    d_sum = tf.summary.histogram("d", D_real)
    d__sum = tf.summary.histogram("d_", D_fake)
    G_sum = tf.summary.image("G", G)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.zeros_like(D_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.ones_like(D_fake)))

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

    d_loss = d_loss_real + d_loss_fake

    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    g_sum = tf.summary.merge([z_sum, d__sum,
                              G_sum, d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge(
        [z_sum, d_sum, d_loss_real_sum, d_loss_sum])
    writer = tf.summary.FileWriter("./logs", tf.get_default_graph())
    writer.flush()

    saver = tf.train.Saver()

    counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        num_batches = int(dataset.num_batches(batch_size))
        for i in range(num_batches):
            X_batch, y_batch = dataset.next_batch(batch_size)
            Z_sample = sample_Z(batch_size, Z_dim)

            # Update D network
            _, summary_str = sess.run([d_optim, d_sum], feed_dict={X: X_batch, Z: Z_sample, y: y_batch})
            writer.add_summary(summary_str, counter)

            # Update G network
            _, summary_str = sess.run([g_optim, g_sum], feed_dict={Z: Z_sample, y: y_batch})
            writer.add_summary(summary_str, counter)

            # # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            # _, summary_str = sess.run([g_optim, g_sum], feed_dict={Z: Z_sample, y: y_batch})
            # writer.add_summary(summary_str, counter)
            with sess.as_default():
                errD_fake = d_loss_fake.eval({Z: Z_sample, y: y_batch})
                errD_real = d_loss_real.eval({X: X_batch, y: y_batch})
                errG = g_loss.eval({Z: Z_sample, y: y_batch})

            counter += 1
            print("Epoch: {:2d} {:4d}/{:4d} time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(epoch, i, num_batches,
                                                                                                  time.time() - start_time,
                                                                                                  errD_fake + errD_real,
                                                                                                  errG))

            if np.mod(counter, 100) == 1:
                try:
                    samples = sess.run(sampler, feed_dict={
                        Z: Z_sample,
                        y: y_batch
                    })
                    d_loss_val, g_loss_val = sess.run(
                        [d_loss, g_loss],
                        feed_dict={
                            Z: Z_sample,
                            X: X_batch,
                            y: y_batch
                        },
                    )
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, i))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss_val, g_loss_val))
                except Exception as e:
                    print("pic saving error:")
                    print(e)
                    raise e

            if np.mod(counter, 400) == 2:
                save(checkpoint_dir, counter, batch_size, image_size, saver, sess)
                print("saved after {}. iteration".format(counter))

    writer.flush()
    writer.close()
    save(checkpoint_dir, counter, batch_size, image_size, saver, sess)
    print("learning has ended")


if __name__ == '__main__':
    main()
