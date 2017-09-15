import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from python.network_utils import conv_cond_concat


class GeneratorFactory:
    def __init__(self, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim, scope_name='generator'):
        self.c_dim = c_dim
        self.gf_dim = gf_dim
        self.gfc_dim = gfc_dim
        self.batch_size = batch_size
        self.image_size = image_size
        self.y_dim = y_dim
        self.scope_name = scope_name

    def create(self, z, y, is_training=True, reuse=False):
        with tf.variable_scope(self.scope_name) as scope:
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

            # first argument is where to apply these
            with arg_scope([layers.conv2d, layers.conv2d_transpose, layers.fully_connected],
                           normalizer_fn=layers.batch_norm,
                           activation_fn=slim.nn.relu,
                           normalizer_params=batch_norm_params,
                           weights_initializer=layers.xavier_initializer(uniform=False),
                           biases_initializer=tf.constant_initializer(0.0)
                           ):
                # taken from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
                s_h, s_w = self.image_size[1], self.image_size[0]
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat([z, y], 1)

                h0 = slim.fully_connected(z,
                                          num_outputs=self.gfc_dim,
                                          scope='g_h0_lin',
                                          )

                h0 = tf.concat([h0, y], 1)

                h1 = slim.fully_connected(h0,
                                          num_outputs=self.gf_dim * 2 * s_h4 * s_w4,
                                          scope='g_h1_lin',
                                          )

                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = slim.conv2d_transpose(h1,
                                           num_outputs=self.gf_dim * 2,
                                           scope='g_h2',
                                           kernel_size=[5, 5],
                                           stride=2,
                                           )

                h2 = conv_cond_concat(h2, yb)

                h3 = slim.conv2d_transpose(h2,
                                           num_outputs=self.c_dim,
                                           scope='g_h3',
                                           kernel_size=[5, 5],
                                           stride=2,
                                           normalizer_fn=None
                                           )

                return tf.identity(h3, 'generator')
