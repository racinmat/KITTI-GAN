import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework.ops import GraphKeys
import tensorflow.contrib.slim as slim
from python.network_utils import sample_z_uniform, save_images, image_manifold_size
from python.neural_network.AbstractNetwork import AbstractNetwork
import math
import tensorflow.contrib.layers as layers
from tensorflow.contrib.slim import arg_scope
from python.network_utils import conv_cond_concat, lrelu


class GanNetworkSlim(AbstractNetwork):
    def __init__(self, checkpoint_dir, name='gan_slim', config=None):
        super().__init__(checkpoint_dir, name, config)

    def build_model(self, image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1, df_dim,
                    dfc_dim):
        g = tf.Graph()

        self.y_dim = y_dim
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.dfc_dim = dfc_dim
        self.gfc_dim = gfc_dim
        self.c_dim = c_dim
        self.image_size = image_size
        self.batch_size = batch_size
        self.z_dim = z_dim

        with g.as_default():
            x = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], c_dim], name='x')
            y = tf.placeholder(tf.float32, shape=[batch_size, y_dim], name='y')
            z = tf.placeholder(tf.float32, shape=[batch_size, z_dim], name='z')

            G = self.create_generator(z, y, self.generator_scope_name, is_training=True, reuse=False)
            sampler = G

            D_real, D_logits_real = self.create_discriminator(x, y, self.discriminator_scope_name, is_training=True, reuse=False)
            D_fake, D_logits_fake = self.create_discriminator(G, y, self.discriminator_scope_name, is_training=True, reuse=True)

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

            d_vars = slim.get_variables(scope=self.discriminator_scope_name, collection=GraphKeys.TRAINABLE_VARIABLES)
            g_vars = slim.get_variables(scope=self.generator_scope_name, collection=GraphKeys.TRAINABLE_VARIABLES)

            d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

            # now I add all trainable variables to summary
            trainable_vars = slim.get_variables(collection=GraphKeys.TRAINABLE_VARIABLES)
            for variable in trainable_vars:
                name = variable.name.split(':', 1)[0]
                tf.summary.histogram(name, variable)

            sess = tf.Session(graph=g, config=self.config)
            sess.run(tf.global_variables_initializer())

            summ = tf.summary.merge_all()
            self.saver = tf.train.Saver()
            self.graph = g
            self.d_optim = d_optim
            self.g_optim = g_optim
            self.summ = summ
            self.sampler = sampler
            self.sess = sess
            self.d_loss_fake = d_loss_fake
            self.d_loss_real = d_loss_real
            self.x = x
            self.y = y
            self.z = z
            self.d_loss = d_loss
            self.g_loss = g_loss

    def train(self, data_set, logs_dir, epochs, sample_dir):
        if not os.path.exists(os.path.dirname(logs_dir)):
            os.makedirs(os.path.dirname(logs_dir))
            print("creating logs dir for training: " + logs_dir)

        with self.graph.as_default():
            writer = tf.summary.FileWriter(os.path.join(logs_dir, self.name), tf.get_default_graph())

            counter = 0
            start_time = time.time()

            print("Starting to learn for {} epochs.".format(epochs))
            for epoch in range(epochs):
                num_batches = int(data_set.num_batches(self.batch_size))
                for i in range(num_batches):
                    x_batch, y_batch = data_set.next_batch(self.batch_size)
                    z_batch = sample_z_uniform(self.batch_size, self.z_dim)

                    # Update D network
                    _, errD_fake, errD_real = self.sess.run([self.d_optim, self.d_loss_fake, self.d_loss_real],
                                                            feed_dict={
                                                                self.x: x_batch,
                                                                self.y: y_batch,
                                                                self.z: z_batch,
                                                            })

                    # Update G network
                    _ = self.sess.run([self.g_optim], feed_dict={
                        self.x: x_batch,
                        self.y: y_batch,
                        self.z: z_batch,
                    })

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, errG = self.sess.run([self.g_optim, self.g_loss], feed_dict={
                        self.x: x_batch,
                        self.y: y_batch,
                        self.z: z_batch,
                    })

                    # run summary of all
                    summary_str = self.sess.run(self.summ, feed_dict={
                        self.x: x_batch,
                        self.y: y_batch,
                        self.z: z_batch,
                    })
                    writer.add_summary(summary_str, counter)

                    counter += 1
                    summary_string = "Epoch: {:2d} {:2d}/{:2d} counter: {:3d} time: {:4.4f}, d_loss: {:.6f}, g_loss: {:.6f}"
                    print(summary_string.format(epoch, i, num_batches, counter, time.time() - start_time,
                                                errD_fake + errD_real,
                                                errG))

                    if np.mod(counter, 100) == 1:
                        try:
                            samples = self.sess.run(self.sampler, feed_dict={
                                self.y: y_batch,
                                self.z: z_batch,
                            })
                            d_loss_val, g_loss_val = self.sess.run([self.d_loss, self.g_loss], feed_dict={
                                self.x: x_batch,
                                self.y: y_batch,
                                self.z: z_batch,
                            })
                            save_images(samples, image_manifold_size(samples.shape[0]),
                                        './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, i))
                            print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss_val, g_loss_val))
                        except Exception as e:
                            print("pic saving error:")
                            print(e)
                            raise e

                    if np.mod(counter, 800) == 2:
                        self.save(counter)
                        print("saved after {}. iteration".format(counter))

            writer.flush()
            writer.close()
            self.save(counter)

    def generate(self, features, samples_dir, suffix):
        z = self.graph.get_tensor_by_name('z:0')
        y = self.graph.get_tensor_by_name('y:0')

        # samples
        samples_num = 1
        for idx in range(samples_num):
            image_frame_dim = int(math.ceil(self.batch_size ** .5))
            z_sample = sample_z_uniform(self.batch_size, self.z_dim)

            samples = self.sess.run(self.sampler, feed_dict={z: z_sample, y: features})

            if not os.path.exists(os.path.dirname(samples_dir)):
                os.makedirs(os.path.dirname(samples_dir))

            save_images(samples, [image_frame_dim, image_frame_dim],
                        '{}/test_{}_{:d}.png'.format(samples_dir, suffix, idx))
            print("images saved to dir {}".format(samples_dir))

    def create_discriminator(self, x, y, scope_name, is_training=True, reuse=False):
        with tf.variable_scope(scope_name) as scope:
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

    def create_generator(self, z, y, scope_name, is_training=True, reuse=False):
        with tf.variable_scope(scope_name) as scope:
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
                                          activation_fn=slim.nn.relu,
                                          )

                h0 = tf.concat([h0, y], 1)

                h1 = slim.fully_connected(h0,
                                          num_outputs=self.gf_dim * 2 * s_h4 * s_w4,
                                          scope='g_h1_lin',
                                          activation_fn=slim.nn.relu,
                                          )

                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = slim.conv2d_transpose(h1,
                                           num_outputs=self.gf_dim * 2,
                                           scope='g_h2',
                                           kernel_size=[5, 5],
                                           stride=2,
                                           activation_fn=slim.nn.relu,
                                           )

                h2 = conv_cond_concat(h2, yb)

                h3 = slim.conv2d_transpose(h2,
                                           num_outputs=self.c_dim,
                                           scope='g_h3',
                                           kernel_size=[5, 5],
                                           stride=2,
                                           normalizer_fn=None,
                                           activation_fn=slim.nn.sigmoid
                                           )

                return h3
