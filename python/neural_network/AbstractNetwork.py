import os
import tensorflow as tf
from python.network_utils import sample_z_uniform, save_images
import math


class AbstractNetwork:
    def __init__(self, checkpoint_dir, name):
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

    def build_model(self, image_size, y_dim, batch_size, c_dim, z_dim, gfc_dim, gf_dim, l1_ratio, learning_rate, beta1,
                    df_dim, dfc_dim):
        raise Exception("This is abstract")

    def train(self, data_set, logs_dir, epochs, sample_dir):
        raise Exception("This is abstract")

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
        print(" [*] Loading last checkpoint")

        model_dir_name = 'KITTI_36_32_32'
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir_name)
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            data_file = os.path.join(checkpoint_dir, checkpoint_name)
            self.saver.restore(self.sess, data_file)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
            print(" [*] Success to read {}".format(checkpoint_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
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

            if not os.path.exists(os.path.dirname(samples_dir)):
                os.makedirs(os.path.dirname(samples_dir))

            save_images(samples, [image_frame_dim, image_frame_dim],
                        '{}/test_{}_{:d}.png'.format(samples_dir, suffix, idx))
            print("images saved to dir {}".format(samples_dir))
