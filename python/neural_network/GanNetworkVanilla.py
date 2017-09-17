import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework import ops
import tensorflow.contrib.slim as slim
from python.network_utils import sample_z, save_images, image_manifold_size, conv_cond_concat, lrelu
from python.neural_network.AbstractNetwork import AbstractNetwork


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


class GanNetworkVanilla(AbstractNetwork):
    def __init__(self, checkpoint_dir, name='gan_vanilla'):
        super().__init__(checkpoint_dir, name)

    def build_model(self, image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1, df_dim,
                    dfc_dim):

        # with this, I can use multiple graphs in single process
        g = tf.Graph()
        with g.as_default():
            x = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], c_dim], name='x')
            y = tf.placeholder(tf.float32, shape=[batch_size, y_dim], name='y')
            z = tf.placeholder(tf.float32, shape=[batch_size, z_dim], name='z')
            G = self.generator(z, y, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim)
            D_real, D_logits_real = self.discriminator(x, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=False)
            sampler = G
            D_fake, D_logits_fake = self.discriminator(G, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=True)

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

            d_vars = slim.get_variables(scope='discriminator', collection=ops.GraphKeys.TRAINABLE_VARIABLES)
            g_vars = slim.get_variables(scope='generator', collection=ops.GraphKeys.TRAINABLE_VARIABLES)

            d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

            sess = tf.Session(graph=g)
            sess.run(tf.global_variables_initializer())

            # merge_all zmerguje všechno z obou sítí, je třeba to oddělit., nějak přes ops.get_collection.
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
            self.image_size = image_size
            self.batch_size = batch_size
            self.z_dim = z_dim

    def train(self, data_set, logs_dir, epochs, sample_dir):
        if not os.path.exists(os.path.dirname(logs_dir)):
            os.makedirs(os.path.dirname(logs_dir))
            print("creating logs dir for training: " + logs_dir)

        with self.graph.as_default():
            writer = tf.summary.FileWriter(os.path.join(logs_dir, self.name), tf.get_default_graph())

            saver = tf.train.Saver()

            counter = 0
            start_time = time.time()

            print("Starting to learn for {} epochs.".format(epochs))
            for epoch in range(epochs):
                num_batches = int(data_set.num_batches(self.batch_size))
                for i in range(num_batches):
                    x_batch, y_batch = data_set.next_batch(self.batch_size)
                    z_batch = sample_z(self.batch_size, self.z_dim)

                    # Update D network
                    _, errD_fake, errD_real = self.sess.run([self.d_optim, self.d_loss_fake, self.d_loss_real], feed_dict={
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

    @staticmethod
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

    @staticmethod
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
