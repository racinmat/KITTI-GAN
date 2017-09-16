import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

from python.network_utils import conv_cond_concat, lrelu


class DiscriminatorFactory:
    def __init__(self, image_size, batch_size, y_dim, dfc_dim, df_dim, c_dim, scope_name='discriminator'):
        self.c_dim = c_dim
        self.df_dim = df_dim
        self.dfc_dim = dfc_dim
        self.batch_size = batch_size
        self.image_size = image_size
        self.y_dim = y_dim
        self.scope_name = scope_name

    def create(self, x, y, is_training=True, reuse=False):
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

            with arg_scope([layers.conv2d, layers.conv2d_transpose, layers.fully_connected],
                           normalizer_fn=layers.batch_norm,
                           normalizer_params=batch_norm_params,
                           weights_initializer=layers.xavier_initializer(uniform=False),
                           biases_initializer=tf.constant_initializer(0.0)
                           ):
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(x, yb)

                h0 = slim.conv2d(x,
                                 num_outputs=self.c_dim + self.y_dim,
                                 scope='d_h0_conv',
                                 kernel_size=[5, 5],
                                 stride=[2, 2],
                                 normalizer_fn=None,
                                 activation_fn=lrelu,
                                 )

                h0 = conv_cond_concat(h0, yb)

                # not having bias variables here because of bias adding in batch normalization, see: https://stackoverflow.com/questions/46256747/can-not-use-both-bias-and-batch-normalization-in-convolution-layers
                h1 = slim.conv2d(h0,
                                 num_outputs=self.df_dim + self.y_dim,
                                 scope='d_h1_conv',
                                 kernel_size=[5, 5],
                                 stride=[2, 2],
                                 activation_fn=lrelu
                                 )

                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = tf.concat([h1, y], 1)

                h2 = slim.fully_connected(h1,
                                          num_outputs=self.dfc_dim,
                                          scope='d_h2_lin',
                                          activation_fn=lrelu,
                                          )
                h2 = tf.concat([h2, y], 1)

                h3 = slim.fully_connected(h2,
                                          num_outputs=1,
                                          scope='d_h3_lin',
                                          normalizer_fn=None,
                                          activation_fn=None
                                          )

                return tf.nn.sigmoid(h3), h3
