import os
import numpy as np
import scipy.misc
import tensorflow as tf
import time
from tensorflow.python.framework.ops import GraphKeys
import tensorflow.contrib.slim as slim


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


class NeuralNetwork:
    def __init__(self, session, input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
        """

        Args:
          session: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.session = session
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        if self.dataset_name == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
            self.c_dim = self.data_X[0].shape[-1]
        else:
            self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
            imreadImg = imread(self.data[0]);
            if len(imreadImg.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

        self.build_model()


    def discriminator(self, x, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=False):
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

    def generator(self, z, y, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim):
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

    def sample_z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def merge(self, images, size):
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

    def model_dir(self, batch_size, image_size):
        return "{}_{}_{}_{}".format(
            'KITTI', batch_size,
            image_size[1], image_size[0])

    def save(self, checkpoint_dir, step, batch_size, image_size, saver, sess, model_name):
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir(batch_size, image_size))

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=step)

    def imsave(self, images, size, path):
        image = np.squeeze(merge(images, size))
        return scipy.misc.imsave(path, image)

    def inverse_transform(self, images):
        """
        :type images: np.ndarray
        :rtype np.ndarray
        """
        return (images + 1.) / 2.

    def save_images(self, images, size, image_path):
        if not os.path.exists(os.path.dirname(image_path)):
            os.makedirs(os.path.dirname(image_path))

        return imsave(inverse_transform(images), size, image_path)

    def image_manifold_size(self, num_images):
        manifold_h = int(np.floor(np.sqrt(num_images)))
        manifold_w = int(np.ceil(np.sqrt(num_images)))
        assert manifold_h * manifold_w == num_images
        return manifold_h, manifold_w

    def load(self, session, checkpoint_dir):
        import re
        print(" [*] Loading last checkpoint")

        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            data_file = os.path.join(checkpoint_dir, checkpoint_name)
            meta_file = data_file + '.meta'
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(session, data_file)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
            print(" [*] Success to read {}".format(checkpoint_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def build_gan(self, data_set, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1, df_dim, dfc_dim):
        image_size = data_set.get_image_size()
        y_dim = data_set.get_labels_dim()

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

        d_vars = slim.get_variables(scope='discriminator', collection=GraphKeys.TRAINABLE_VARIABLES)
        g_vars = slim.get_variables(scope='generator', collection=GraphKeys.TRAINABLE_VARIABLES)

        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        summ = tf.summary.merge_all()
        return d_optim, g_optim, summ, sampler, sess, d_loss_fake, d_loss_real, x, y, z, d_loss, g_loss, image_size

    def train_network(self, logs_dir, epochs, batch_size, z_dim, sess, d_optim, g_optim, d_loss_fake,
                      d_loss_real, x, y, z, data_set, d_loss, g_loss, summ, sampler, sample_dir, checkpoint_dir,
                      image_size, model_name):
        if not os.path.exists(os.path.dirname(logs_dir)):
            os.makedirs(os.path.dirname(logs_dir))

        writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

        saver = tf.train.Saver()

        counter = 0
        start_time = time.time()

        print("Starting to learn for {} epochs.".format(epochs))
        for epoch in range(epochs):
            num_batches = int(data_set.num_batches(batch_size))
            for i in range(num_batches):
                x_batch, y_batch = data_set.next_batch(batch_size)
                z_batch = sample_z(batch_size, z_dim)

                # Update D network
                _, errD_fake, errD_real = sess.run([d_optim, d_loss_fake, d_loss_real], feed_dict={
                    x: x_batch,
                    y: y_batch,
                    z: z_batch,
                })

                # Update G network
                _ = sess.run([g_optim], feed_dict={
                    x: x_batch,
                    y: y_batch,
                    z: z_batch,
                })

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, errG = sess.run([g_optim, g_loss], feed_dict={
                    x: x_batch,
                    y: y_batch,
                    z: z_batch,
                })

                # run summary of all
                summary_str = sess.run(summ, feed_dict={x: x_batch, z: z_batch, y: y_batch})
                writer.add_summary(summary_str, counter)

                counter += 1
                summary_string = "Epoch: {:2d} {:2d}/{:2d} counter: {:3d} time: {:4.4f}, d_loss: {:.6f}, g_loss: {:.6f}"
                print(summary_string.format(epoch, i, num_batches, counter, time.time() - start_time, errD_fake + errD_real,
                                            errG))

                if np.mod(counter, 100) == 1:
                    try:
                        samples = sess.run(sampler, feed_dict={
                            z: z_batch,
                            y: y_batch
                        })
                        d_loss_val, g_loss_val = sess.run([d_loss, g_loss], feed_dict={
                            z: z_batch,
                            x: x_batch,
                            y: y_batch
                        })
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, i))
                        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss_val, g_loss_val))
                    except Exception as e:
                        print("pic saving error:")
                        print(e)
                        raise e

                if np.mod(counter, 800) == 2:
                    save(checkpoint_dir, counter, batch_size, image_size, saver, sess, model_name)
                    print("saved after {}. iteration".format(counter))

        writer.flush()
        writer.close()
        save(checkpoint_dir, counter, batch_size, image_size, saver, sess, model_name)
