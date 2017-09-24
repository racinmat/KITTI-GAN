import os
import tensorflow as tf
import time

from python.network_utils import sample_z_uniform, save_images, image_manifold_size
import math
import numpy as np


class AbstractNetwork:
    def __init__(self, checkpoint_dir, name, config=None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.graph = None
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
        self.name = name
        self.generator_scope_name = 'generator'
        self.discriminator_scope_name = 'discriminator'
        self.saver = None
        self.y_dim = None
        self.df_dim = None
        self.gf_dim = None
        self.dfc_dim = None
        self.gfc_dim = None
        self.c_dim = None

    def build_model(self, image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1,
                    df_dim, dfc_dim):
        raise Exception("This is abstract")

    def build_empty_model(self, image_size, batch_size, z_dim):
        # this is used for loading with structure
        g = tf.Graph()

        self.image_size = image_size
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.graph = g
        with g.as_default():
            sess = tf.Session(graph=g, config=self.config)
            self.sess = sess

    def train(self, data_set, logs_dir, epochs, sample_dir, train_test_ratios):
        if not os.path.exists(os.path.dirname(logs_dir)):
            os.makedirs(os.path.dirname(logs_dir))
            tf.logging.info("creating logs dir for training: " + logs_dir)

        train, test = data_set.split(train_test_ratios)

        with self.graph.as_default():
            writer = tf.summary.FileWriter(os.path.join(logs_dir, self.name), tf.get_default_graph())

            counter = 0
            start_time = time.time()

            tf.logging.info("Starting to learn for {} epochs.".format(epochs))
            for epoch in range(epochs):
                num_batches = int(train.num_batches(self.batch_size))
                for i in range(num_batches):
                    x_batch, y_batch = train.next_batch(self.batch_size)
                    z_batch = sample_z_uniform(self.batch_size, self.z_dim)
                    x_test_batch, y_test_batch = test.next_batch(self.batch_size)

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
                    tf.logging.info(summary_string.format(epoch, i, num_batches, counter, time.time() - start_time,
                                                errD_fake + errD_real,
                                                errG))

                    if np.mod(counter, 100) == 1:
                        try:
                            samples = self.sess.run(self.sampler, feed_dict={
                                self.y: y_test_batch,
                                self.z: z_batch,
                            })
                            d_loss_val, g_loss_val = self.sess.run([self.d_loss, self.g_loss], feed_dict={
                                self.x: x_test_batch,
                                self.y: y_test_batch,
                                self.z: z_batch,
                            })
                            save_images(samples, image_manifold_size(samples.shape[0]),
                                        '{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, i))
                            tf.logging.info("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss_val, g_loss_val))
                        except Exception as e:
                            tf.logging.info("pic saving error:")
                            tf.logging.info(e)
                            raise e

                    if np.mod(counter, 800) == 2:
                        self.save(counter)
                        tf.logging.info("saved after {}. iteration".format(counter))

            writer.flush()
            writer.close()
            self.save(counter)

    def save(self, counter):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir())

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.name), global_step=counter)

    def model_dir(self):
        return "{}_{}_{}_{}".format(
            'KITTI', self.batch_size,
            self.image_size[1], self.image_size[0])

    def load(self):
        import re
        tf.logging.info(" [*] Loading last checkpoint")

        model_dir_name = self.model_dir()
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir_name)
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            data_file = os.path.join(checkpoint_dir, checkpoint_name)
            self.saver.restore(self.sess, data_file)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
            tf.logging.info(" [*] Success to read {}".format(checkpoint_name))
            return True, counter
        else:
            tf.logging.info(" [*] Failed to find a checkpoint")
            return False, 0

    def load_with_structure(self, sampler_name):
        import re
        tf.logging.info(" [*] Loading last checkpoint")

        model_dir_name = self.model_dir()

        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir_name)
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            with self.graph.as_default():
                checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
                data_file = os.path.join(checkpoint_dir, checkpoint_name)
                meta_file = data_file + '.meta'
                saver = tf.train.import_meta_graph(meta_file)
                saver.restore(self.sess, data_file)
                counter = int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
                self.sampler = self.graph.get_tensor_by_name(sampler_name)
                tf.logging.info(" [*] Success to read {}".format(checkpoint_name))
                return True, counter
        else:
            tf.logging.info(" [*] Failed to find a checkpoint")
            return False, 0

    def generate(self, features, samples_dir, suffix):
        y = self.graph.get_tensor_by_name('y:0')
        z = self.graph.get_tensor_by_name('z:0')

        # samples
        samples_num = 1
        for idx in range(samples_num):
            image_frame_dim = int(math.ceil(self.batch_size ** .5))
            z_sample = sample_z_uniform(self.batch_size, self.z_dim)

            samples = self.sess.run(self.sampler, feed_dict={z: z_sample, y: features})

            if not os.path.exists(samples_dir):
                os.makedirs(samples_dir)

            save_images(samples, [image_frame_dim, image_frame_dim],
                        '{}/test_{}_{:d}.png'.format(samples_dir, suffix, idx))
            tf.logging.info("images saved to dir {}".format(samples_dir))
