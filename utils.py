import numpy as np
import tensorflow as tf
import scipy

flags = tf.app.flags
FLAGS = flags.FLAGS


def read_and_decode_with_labels(serialized_example, image_size=128):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64)
        })
    image_size = FLAGS.image_size
    image = tf.image.decode_image(features['image/encoded'], channels=3)
    orig_height = tf.cast(features['image/height'], tf.int32)
    orig_width = tf.cast(features['image/width'], tf.int32)
    channels = tf.cast(features['image/channels'], tf.int32)
    image = tf.reshape(image, [orig_height, orig_width, channels])
    image = tf.image.resize_images(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) * (2. / 255) - 1.
    label = tf.cast(features['image/class/label'], tf.int32)
    return (image, label)


def sigmoid_kl_with_logits(logits, targets):
    assert isinstance(targets, float)
    if targets in [0., 1.]:
        entropy = 0.
    else:
        entropy = - targets * np.log(targets) - \
            (1. - targets) * np.log(1. - targets)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * targets) - entropy


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img
