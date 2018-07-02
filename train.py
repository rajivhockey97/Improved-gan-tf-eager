import tensorflow as tf
from math import sqrt
import tensorflow.contrib.eager as tfe

from utils import sigmoid_kl_with_logits, save_images, read_and_decode_with_labels

flags = tf.app.flags

FLAGS = flags.FLAGS


def discriminator_loss(class_logits, labels):

    num_classes = 1001
    d_label_smooth = FLAGS.d_label_smooth
    class_loss_weight = 1.
    batch_size = int(class_logits.get_shape()[0])

    assert batch_size == 2 * FLAGS.batch_size

    generated_class_logits = tf.squeeze(tf.slice(class_logits, [0, num_classes - 1], [batch_size, 1]))
    positive_class_logits = tf.slice(class_logits, [0, 0], [batch_size, num_classes - 1])
    mx = tf.reduce_max(positive_class_logits, 1, keepdims=True)
    safe_pos_class_logits = positive_class_logits - mx

    gan_logits = tf.log(tf.reduce_sum(tf.exp(safe_pos_class_logits), 1)) + tf.squeeze(mx) - generated_class_logits
    assert len(gan_logits.get_shape()) == 1

    probs = tf.nn.sigmoid(gan_logits)

    D_on_data = tf.slice(probs, [0], [FLAGS.batch_size])
    D_on_data_logits = tf.slice(gan_logits, [0], [FLAGS.batch_size])
    D_on_G = tf.slice(probs, [FLAGS.batch_size], [FLAGS.batch_size])
    D_on_G_logits = tf.slice(gan_logits, [FLAGS.batch_size], [FLAGS.batch_size])
    d_loss_real = sigmoid_kl_with_logits(D_on_data_logits, 1. - d_label_smooth)
    class_logits = tf.slice(class_logits, [0, 0], [FLAGS.batch_size, num_classes])
    d_loss_class = class_loss_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_logits, labels=tf.to_int64(labels))
    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_on_G_logits, labels=tf.zeros_like(D_on_G_logits))
    d_loss_class = tf.reduce_mean(d_loss_class)
    d_loss_real = tf.reduce_mean(d_loss_real)
    d_loss_fake = tf.reduce_mean(d_loss_fake)
    d_loss = d_loss_real + d_loss_fake + d_loss_class

    tf.contrib.summary.histogram("D_on_data", D_on_data)
    tf.contrib.summary.histogram("D_on_data_logits", D_on_data_logits)
    tf.contrib.summary.histogram("D_on_G", D_on_G)
    tf.contrib.summary.histogram("D_on_G_logits", D_on_G_logits)
    tf.contrib.summary.histogram("class_logits", class_logits)
    tf.contrib.summary.histogram("d_loss_real", d_loss_real)
    tf.contrib.summary.histogram("d_loss_fake", d_loss_fake)
    tf.contrib.summary.histogram("d_loss_class", d_loss_class)
    tf.contrib.summary.histogram("d_loss", d_loss)
    return d_loss, D_on_G_logits


def generator_loss(D_on_G_logits, generator_target_prob):
    g_loss = sigmoid_kl_with_logits(D_on_G_logits, FLAGS.generator_target_prob)
    g_loss = tf.reduce_mean(g_loss)
    tf.contrib.summary.histogram("g_loss", g_loss)
    return g_loss


def train_one_epoch(dataset, generator, discriminator, generator_optimizer,
                    discriminator_optimizer, batch_size, step_counter,
                    log_interval, z_dim, epoch, device):

    total_generator_loss = 0.0
    total_discriminator_loss = 0.0

    with tf.device(device):
        tf.assign_add(step_counter, 1)
    for (batch_index, (images, labels)) in enumerate(tfe.Iterator(dataset)):
        with tf.contrib.summary.record_summaries_every_n_global_steps(log_interval, global_step=step_counter):
            noise = tf.random_uniform(shape=[batch_size, z_dim],
                                             minval=-1.,
                                             maxval=1.,
                                             seed=batch_index)
            with tf.GradientTape(persistent=True) as grad_tape:
                G, zs = generator(noise)
                joint = tf.concat([images, G], 0)
                class_logits = discriminator(joint)
                discriminator_loss_val, D_on_G_logits = discriminator_loss(class_logits, labels)
                total_discriminator_loss += discriminator_loss_val
                generator_loss_val = generator_loss(D_on_G_logits, FLAGS.generator_target_prob)
                total_generator_loss += generator_loss_val
        generator_grad = grad_tape.gradient(generator_loss_val, generator.variables)
        discriminator_grad = grad_tape.gradient(discriminator_loss_val, discriminator.variables)
        generator_optimizer.apply_gradients(zip(generator_grad, generator.variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_grad, discriminator.variables))

        image_sum = False
        if image_sum:
            G_means = tf.reduce_mean(G, 0, keepdims=True)
            G_vars = tf.reduce_mean(tf.square(G - G_means), 0, keepdims=True)
            tf.contrib.summary.scalar('G_means', G_means)
            tf.contrib.summary.scalar('G_vars', G_vars)
            image_means = tf.reduce_mean(images, 0, keepdims=True)
            image_vars = tf.reduce_mean(tf.square(images - image_means), 0, keepdims=True)
            tf.contrib.summary.scalar('image_means', image_means)
            tf.contrib.summary.scalar('image_vars', image_vars)

        if log_interval and batch_index > 0 and batch_index % log_interval == 0:
            print('Batch #%d\tAverage Generator Loss: %.6f\t'
                  'Average Discriminator Loss: %.6f' %
                  (batch_index, total_generator_loss / batch_index,
                  total_discriminator_loss / batch_index))

        if batch_index > 0 and batch_index % FLAGS.save_interval == 0:
            save_images(G, [sqrt(FLAGS.batch_size), sqrt(FLAGS.batch_size)],
                        FLAGS.sample_dir + '/train_epoch_%s_batch_%s.png'
                        % (epoch, batch_index))
