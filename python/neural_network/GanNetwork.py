import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework import ops
import tensorflow.contrib.slim as slim
from python.network_utils import generator, discriminator, sample_z, save_images, image_manifold_size, save


class GanNetwork:
    def __init__(self, scope_name='GAN'):
        self.d_optim = None
        self.g_optim = None
        self.summ = None
        self.sampler = None
        self.sess = None
        self.d_loss_fake = None
        self.d_loss_real = None
        self.x = None
        self.y = None
        self.z = None
        self.d_loss = None
        self.g_loss = None
        self.image_size = None
        self.batch_size = None
        self.z_dim = None
        self.data_set = None
        self.scope_name = scope_name

    def build_model(self, data_set, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1, df_dim,
                    dfc_dim):
        image_size = data_set.get_image_size()
        y_dim = data_set.get_labels_dim()

        with tf.variable_scope(self.scope_name):
            x = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], c_dim], name='x')
            y = tf.placeholder(tf.float32, shape=[batch_size, y_dim], name='y')
            z = tf.placeholder(tf.float32, shape=[batch_size, z_dim], name='z')
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

            d_vars = slim.get_variables(scope=self.scope_name + '/discriminator', collection=ops.GraphKeys.TRAINABLE_VARIABLES)
            g_vars = slim.get_variables(scope=self.scope_name + '/generator', collection=ops.GraphKeys.TRAINABLE_VARIABLES)

            d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            # merge_all zmerguje všechno z obou sítí, je třeba to oddělit., nějak přes ops.get_collection.
            summ = tf.summary.merge_all()
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
            self.data_set = data_set

    def train(self, logs_dir, epochs, sample_dir, checkpoint_dir,
              model_name):

        if not os.path.exists(os.path.dirname(logs_dir)):
            os.makedirs(os.path.dirname(logs_dir))

        writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

        saver = tf.train.Saver()

        counter = 0
        start_time = time.time()

        print("Starting to learn for {} epochs.".format(epochs))
        for epoch in range(epochs):
            num_batches = int(self.data_set.num_batches(self.batch_size))
            for i in range(num_batches):
                x_batch, y_batch = self.data_set.next_batch(self.batch_size)
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
                    save(checkpoint_dir, counter, self.batch_size, self.image_size, saver, self.sess, model_name)
                    print("saved after {}. iteration".format(counter))

        writer.flush()
        writer.close()
        save(checkpoint_dir, counter, self.batch_size, self.image_size, saver, self.sess, model_name)