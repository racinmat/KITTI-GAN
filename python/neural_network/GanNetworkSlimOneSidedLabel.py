import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework.ops import GraphKeys
import tensorflow.contrib.slim as slim
from python.network_utils import sample_z_uniform, save_images, image_manifold_size
from python.neural_network.AbstractNetwork import AbstractNetwork
from python.neural_network.DiscriminatorFactory import DiscriminatorFactory
from python.neural_network.GeneratorFactory import GeneratorFactory
import math


class GanNetworkSlimOneSidedLabel(AbstractNetwork):
    def __init__(self, checkpoint_dir, name='gan_slim'):
        super().__init__(checkpoint_dir, name)

    def build_model(self, image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1, df_dim,
                    dfc_dim, smooth = 0):
        g = tf.Graph()
        with g.as_default():
            x = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], c_dim], name='x')
            y = tf.placeholder(tf.float32, shape=[batch_size, y_dim], name='y')
            z = tf.placeholder(tf.float32, shape=[batch_size, z_dim], name='z')

            G_factory = GeneratorFactory(image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim,
                                         self.generator_scope_name)
            D_factory = DiscriminatorFactory(image_size, batch_size, y_dim, dfc_dim, df_dim, c_dim,
                                             self.discriminator_scope_name)
            G = G_factory.create(z, y)
            sampler = G

            D_real, D_logits_real = D_factory.create(x, y, reuse=False)
            D_fake, D_logits_fake = D_factory.create(G, y, reuse=True)

            tf.summary.histogram("z", z)
            tf.summary.histogram("d_real", D_real)
            tf.summary.histogram("d_fake", D_fake)
            tf.summary.image("g", G)

            # smmothing is applied only on discriminator: https://medium.com/towards-data-science/gan-introduction-and-implementation-part1-implement-a-simple-gan-in-tf-for-mnist-handwritten-de00a759ae5c
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_real) * (1 - smooth)))
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

            sess = tf.Session(graph=g)
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
            self.image_size = image_size
            self.batch_size = batch_size
            self.z_dim = z_dim

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
