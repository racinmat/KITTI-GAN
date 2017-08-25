from python.neural_network.Dataset import DataSet
import tensorflow.contrib.slim as slim
from python.network_utils import *
import time
import tensorflow as tf
import numpy as np
import os
import pickle
from python.neural_network.train_gan import load_data
from tensorflow.python.framework.ops import GraphKeys

def main():
    data_dir = 'data/extracted'
    drives = [
        'drive_0009_sync',
        'drive_0015_sync',
        'drive_0023_sync',
        'drive_0032_sync',
    ]

    input_prefix = 'tracklets_photos_normalized_'
    resolution = (32, 32)

    data_set = load_data(resolution, drives, input_prefix, data_dir)

    # batch_size = 64
    batch_size = 36
    z_dim = 100
    image_size = data_set.get_image_size()
    y_dim = data_set.get_labels_dim()

    l1_ratio = 100
    epochs = 5000
    gf_dim = 64  # (optional) Dimension of gen filters in first conv layer.
    df_dim = 64  # (optional) Dimension of discrim filters in first conv layer.
    gfc_dim = 1024  # (optional) Dimension of gen units for for fully connected layer.
    dfc_dim = 1024  # (optional) Dimension of discrim units for fully connected layer.
    c_dim = 3  # (optional) Dimension of image color. For grayscale input, set to 1, for colors, set to 3.
    learning_rate = 0.0002  # Learning rate of for adam
    beta1 = 0.5  # Momentum term of adam
    current_time = str(int(time.time()))
    sample_dir = os.path.join('samples', current_time)  # Directory name to save the image samples
    checkpoint_dir = os.path.join('checkpoint', current_time)  # Directory name to save the checkpoints
    logs_dir = os.path.join('logs', current_time)
    model_name = 'CGAN.model'

    x = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], c_dim], name='x')
    y = tf.placeholder(tf.float32, shape=[batch_size, y_dim], name='y')
    z = tf.placeholder(tf.float32, shape=[batch_size, z_dim], name='z')
    G = generator(z, y, image_size, batch_size, y_dim, gfc_dim, gf_dim, c_dim)
    D_real, D_logits_real = discriminator(x, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=False)
    sampler = G
    D_fake, D_logits_fake = discriminator(G, y, batch_size, y_dim, c_dim, df_dim, dfc_dim, reuse=True)

    z_sum = tf.summary.histogram("z", z)
    d_real_sum = tf.summary.histogram("d_real", D_real)
    d_fake_sum = tf.summary.histogram("d_fake", D_fake)
    g_sum = tf.summary.image("g", G)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.zeros_like(D_fake)))
    # g_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.ones_like(D_fake)))

    # For generator we use traditional GAN objective as well as L1 loss
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.ones_like(D_fake))) + \
               l1_ratio * tf.reduce_mean(tf.abs(G - x))  # This optimizes the generator.

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

    d_loss = d_loss_real + d_loss_fake

    d_loss_sum = tf.summary.scalar("d_loss", d_loss)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)

    d_vars = slim.get_variables(scope='discriminator', collection=GraphKeys.TRAINABLE_VARIABLES)
    g_vars = slim.get_variables(scope='generator', collection=GraphKeys.TRAINABLE_VARIABLES)

    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    g_sum = tf.summary.merge([z_sum, d_fake_sum, g_sum, d_loss_fake_sum, g_loss_sum])
    d_real_sum = tf.summary.merge([z_sum, d_real_sum, d_loss_real_sum, d_loss_sum])

    summ = tf.summary.merge_all()

    if not os.path.exists(os.path.dirname(logs_dir)):
        os.makedirs(os.path.dirname(logs_dir))

    writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

    saver = tf.train.Saver()

    counter = 0
    start_time = time.time()

    print("Starting to learn for {} epochs with name {}.".format(epochs, current_time))
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
    print("learning has ended")


if __name__ == '__main__':
    main()
