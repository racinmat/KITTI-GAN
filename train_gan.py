import matplotlib

# http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pickle
from math import pi
from Dataset import DataSet

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
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


data_dir = 'data/extracted'
sizes_x = np.empty((1, 0))
sizes_y = np.empty((1, 0))
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
image_size = data[0]['y'].shape
y_dim = len(data[0]['x'])
h_dim = 128

gf_dim = 64  # (optional) Dimension of gen filters in first conv layer.
df_dim = 64  # (optional) Dimension of discrim filters in first conv layer.
gfc_dim = 1024  # (optional) Dimension of gen units for for fully connected layer.
dfc_dim = 1024  # (optional) Dimension of discrim units for fully connected layer.
c_dim = 1  # (optional) Dimension of image color. For grayscale input, set to 1, for colors, set to 3.


# """ Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, image_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
#
# D_W1 = tf.Variable(xavier_init([image_dim + features_dim, h_dim]))
# D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
#
# D_W2 = tf.Variable(xavier_init([h_dim, 1]))
# D_b2 = tf.Variable(tf.zeros(shape=[1]))
#
# theta_D = [D_W1, D_W2, D_b1, D_b2]


def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


# """ Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
#
# G_W1 = tf.Variable(xavier_init([Z_dim + features_dim, h_dim]))
# G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
#
# G_W2 = tf.Variable(xavier_init([h_dim, image_dim]))
# G_b2 = tf.Variable(tf.zeros(shape=[image_dim]))
#
# theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z, y):
    # inputs = tf.concat(axis=1, values=[z, y])
    # G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    # G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    # G_prob = tf.nn.sigmoid(G_log_prob)
    #
    # return G_prob

    g_bn0 = batch_norm(name='g_bn0')
    g_bn1 = batch_norm(name='g_bn1')
    g_bn2 = batch_norm(name='g_bn2')

    # taken from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    s_h, s_w = image_size[1], image_size[0]
    s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
    s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

    # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
    yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
    z = concat([z, y], 1)

    h0 = tf.nn.relu(
        g_bn0(linear(z, gfc_dim, 'g_h0_lin')))
    h0 = concat([h0, y], 1)

    h1 = tf.nn.relu(g_bn1(
        linear(h0, gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
    h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, gf_dim * 2])

    h1 = conv_cond_concat(h1, yb)

    h2 = tf.nn.relu(g_bn2(deconv2d(h1,
                                   [batch_size, s_h2, s_w2, gf_dim * 2], name='g_h2')))
    h2 = conv_cond_concat(h2, yb)

    return tf.nn.sigmoid(
        deconv2d(h2, [batch_size, s_h, s_w, c_dim], name='g_h3'))


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(resolution[0], resolution[1]), cmap='Greys_r')

    return fig


G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
rz_treshold = 5 * pi / 180

for it in range(1000000):
    if it % 1000 == 0:
        n_sample = 16

        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        y_sample[:, 0] = 0  # rz
        y_sample[:, 1] = 1  # h/w ratio
        y_sample[:, 2] = 1  # l/w ratio
        y_sample[:, 3] = 30  # distance
        y_sample[:, 4] = 32  # image size x
        y_sample[:, 5] = 32  # image size y

        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, y_mb = dataset.next_batch(batch_size)

    Z_sample = sample_Z(batch_size, Z_dim)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y: y_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y: y_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
