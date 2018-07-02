#!/usr/bin/env
import os
import glob
import time
import pprint
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from discriminator import Discriminator
from generator import Generator
from train import train_one_epoch
from utils import read_and_decode_with_labels

data_dir = "/home/constantine/train-00000-of-01024"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("epoch", 1, "Epoch to train")
flags.DEFINE_integer("batch_size", 8, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use")
flags.DEFINE_integer("z_dim", 100, "Dimension of input noise to the generator")
flags.DEFINE_integer(
    "df_dim", 64, "Dimension of discrim filters in first conv layer")
flags.DEFINE_integer(
    "gf_dim", 64, "Dimension of gen filters in first conv layer")
flags.DEFINE_integer("log_interval", 100,
                     "Number of batches between logging and writing summaries")
flags.DEFINE_integer("save_interval", 100,
                     "Number of batches to wait before saving generated images")
flags.DEFINE_float("discriminator_learning_rate",
                   0.0004, "Learning rate of for adam")
flags.DEFINE_float("generator_learning_rate", 0.0004,
                   "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "First Momentum term of adam")
flags.DEFINE_float("beta2", 0.9, "Second Momentum term of adam")
flags.DEFINE_float("d_label_smooth", 0.25,
                   "Label smoothing for the positive targets of discriminator")
flags.DEFINE_float("generator_target_prob", 1.,
                   "Generator target prob for sigmoid_kl_with_logits")
flags.DEFINE_string("checkpoint_dir", "checkpoint",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples",
                    "Directory name to save the image samples [samples]")
flags.DEFINE_string("summary_dir", "summaries",
                    "Directory name to write TensorBoard summaries")
flags.DEFINE_boolean("is_train", False, "True for training")
flags.DEFINE_boolean("model_summary", False, "Prints keras model summary")
flags.DEFINE_boolean(
    "no_gpu", False, "Disable Gpu usage even if GPU is available")


def main(_):

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    filenames = glob.glob(data_dir)

    (device, data_format) = ('/gpu:0', 'channels_first')
    if FLAGS.no_gpu or tfe.num_gpus() <= 0:
        (device, data_format) = ('/cpu:0', 'channels_last')
    print('Using device %s, and data format %s.' % (device, data_format))

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    model_objects = {
        'generator': Generator(data_format),
        'discriminator': Discriminator(data_format),
        'generator_optimizer': tf.train.AdamOptimizer(FLAGS.generator_learning_rate, FLAGS.beta1, FLAGS.beta2),
        'discriminator_optimizer': tf.train.AdamOptimizer(FLAGS.discriminator_learning_rate, FLAGS.beta1, FLAGS.beta2),
        'step_counter': tf.train.get_or_create_global_step()
    }

    summary_writer = tf.contrib.summary.create_file_writer(FLAGS.summary_dir,
                                                           flush_millis=1000)

    checkpoint = tfe.Checkpoint(**model_objects)
    checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
    latest_cpkt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if latest_cpkt:
        print('Using latest checkpoint at ' + latest_cpkt)
    checkpoint.restore(latest_cpkt)

    dataset = tf.data.TFRecordDataset(
        filenames).map(read_and_decode_with_labels)
    dataset = dataset.shuffle(10000).apply(
        tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size))

    with tf.device(device):
        for epoch in range(FLAGS.epoch):
            start = time.time()
            with summary_writer.as_default():
                train_one_epoch(dataset=dataset, batch_size=FLAGS.batch_size, log_interval=FLAGS.log_interval,
                                z_dim=FLAGS.z_dim, device=device, epoch=epoch, **model_objects)
            end = time.time()
            checkpoint.save(checkpoint_prefix)
            print('\nTrain time for epoch #%d (step %d): %f' %
                  (checkpoint.save_counter.numpy(),
                   checkpoint.step_counter.numpy(),
                   end - start))


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.app.run(main=main)
