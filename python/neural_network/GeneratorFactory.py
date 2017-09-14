import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope


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


class GeneratorFactory:
    """GeneratorFactory setup.

    Args:
      images: A float32 scalar Tensor of real images from one domain
      scope: name scope
      extra_layers: boolean whether use conv5 layer (1 x 1 x 100 dim) or not
      reuse: reuse flag

    Returns:
      A float32 scalar Tensor of generated images from one domain to another domain
    """

    def __init__(self, z, y, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim):
        self.c_dim = c_dim
        self.gf_dim = gf_dim
        self.gfc_dim = gfc_dim
        self.batch_size = batch_size
        self.image_size = image_size
        self.y = y
        self.z = z
        self.y_dim = y_dim

    def create(self, is_training=True, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            batch_norm_params = {
                # 'decay': 0.999,
                'decay': 0.9,  # also known as momentum, they are the same
                'updates_collections': None,
                # 'epsilon': 0.001,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training,
                'scope': 'batch_norm',
            }
            with arg_scope([layers.conv2d, layers.conv2d_transpose],
                           kernel_size=[4, 4],
                           stride=[2, 2],
                           normalizer_fn=layers.batch_norm,
                           normalizer_params=batch_norm_params) as arg_sc:
                with tf.variable_scope("generator"):
                    g_bn2 = BatchNorm(name='g_bn2')

                    # taken from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
                    s_h, s_w = self.image_size[1], self.image_size[0]
                    s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                    s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                    yb = tf.reshape(self.y, [self.batch_size, 1, 1, self.y_dim])
                    z = tf.concat([self.z, self.y], 1)

                    h0 = self.linear(self.z, self.gfc_dim, scope='g_h0_lin', activation_fn=slim.nn.relu, arg_sc=arg_sc)

                    h0 = tf.concat([h0, y], 1)

                    h1 = self.linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')
                    h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                    h1 = self.conv_cond_concat(h1, yb)

                    h2 = tf.nn.relu(
                        g_bn2(self.deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], scope='g_h2')))
                    h2 = self.conv_cond_concat(h2, yb)

                    h3 = tf.nn.sigmoid(self.deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], scope='g_h3'))

                    return tf.identity(h3, 'generator')

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def conv_cond_concat(self, x, y):
        """Concatenate conditioning vector on feature map axis."""
        # this propagates labels to other layers of neural network, probably according to https://arxiv.org/pdf/1611.01455.pdf
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        return tf.concat([
            x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

    def conv2d(self, input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", biases)

            return conv

    def deconv2d(self, input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, scope="deconv2d", bias_start=0.0):
        deconv = slim.conv2d_transpose(input_,
                                       output_shape,
                                       scope=scope,
                                       weights_initializer=layers.xavier_initializer(uniform=False),
                                       biases_initializer=tf.constant_initializer(bias_start),
                                       stride=[d_h, d_w])

        return deconv
        # with tf.variable_scope(scope):
        #     # filter : [height, width, output_channels, in_channels]
        #     w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
        #                         initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        #
        #     deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
        #                                     strides=[1, d_h, d_w, 1])
        #
        #     biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        #     deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        #
        #     tf.summary.histogram("weights", w)
        #     tf.summary.histogram("biases", biases)
        #
        #     if with_w:
        #         return deconv, w, biases
        #     else:
        #         return deconv

    def lrelu(self, x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x, name=name)

    def linear(self, input_, output_size, scope='linear', bias_start=0.0, activation_fn=slim.nn.relu,
               arg_sc={}):
        with arg_scope(arg_sc):
            output = slim.fully_connected(input_,
                                          output_size,
                                          scope=scope,
                                          activation_fn=activation_fn,
                                          weights_initializer=layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(bias_start))

            return output
            # with tf.variable_scope(scope, reuse=True):
            #     w = tf.get_variable('weights')
            #     bias = tf.get_variable('biases')
            #     tf.summary.histogram("weights", w)
            #     tf.summary.histogram("biases", bias)
            #     return output
