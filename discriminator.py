import tensorflow as tf

layers = tf.keras.layers
seed = 1234
TruncatedNormal = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
Constant = tf.keras.initializers.Constant(value=0.0)
RandomNormal = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=seed)

flags = tf.app.flags

FLAGS = flags.FLAGS


class _Minibatch_block(tf.keras.Model):
    def __init__(self,
                 n_kernels,
                 dim_per_kernel,
                 name=''):
        super(_Minibatch_block, self).__init__(name='')
        self.n_kernels = n_kernels
        self.dim_per_kernel = dim_per_kernel
        self.batch_size = 2*FLAGS.batch_size

        def mask(batch_size):
            big = tf.zeros((batch_size, batch_size), dtype='float32')
            big += tf.eye(batch_size)
            big = tf.expand_dims(big, 1)
            mask = 1. - big
            return mask

        self.mask = mask(self.batch_size)
        self.linear1 = layers.Dense(n_kernels*dim_per_kernel,
                                    kernel_initializer=RandomNormal,
                                    bias_initializer=Constant,
                                    name=name)

    def call(self, inputs):
        x = self.linear1(inputs)
        x = tf.reshape(
            x, (self.batch_size, self.n_kernels, self.dim_per_kernel))
        x = tf.reduce_sum(tf.abs(tf.expand_dims(x, 3) -
                                 tf.expand_dims(tf.transpose(x, [1, 2, 0]), 0)), 2)
        x = tf.exp(-x) * self.mask

        m1, n1, _ = x.get_shape()
        m2, n2, _ = self.mask.get_shape()
        num1 = tf.reduce_sum(
            tf.slice(x, [0, 0, 0], [m1, n1, FLAGS.batch_size]), 2)
        den1 = tf.reduce_sum(
            tf.slice(self.mask, [0, 0, 0], [m2, n2, FLAGS.batch_size]))
        num2 = tf.reduce_sum(tf.slice(x, [0, 0, FLAGS.batch_size], [
                             m1, n1, FLAGS.batch_size]), 2)
        den2 = tf.reduce_sum(
            tf.slice(self.mask, [0, 0, FLAGS.batch_size], [m2, n2, FLAGS.batch_size]))
        f1 = num1/den1
        f2 = num2/den2
        minibatch_block_features = [f1, f2]

        x = tf.concat([inputs] + minibatch_block_features, 1)
        return x


class _Conv2d_block(tf.keras.Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='SAME',
                 data_format=None,
                 name=''):

        name1, name2, name3 = name

        super(_Conv2d_block, self).__init__(name=name1)
        filters1, filters2 = filters

        bn_axis = 1 if data_format == 'channels_first' else -1

        self.conv2a = layers.Conv2D(filters1, kernel_size,
                                    strides=strides, padding=padding,
                                    data_format=data_format,
                                    kernel_initializer=TruncatedNormal,
                                    bias_initializer=Constant, name=name2)
        self.conv2b = layers.Conv2D(filters2, kernel_size,
                                    strides=strides, padding=padding,
                                    data_format=data_format,
                                    kernel_initializer=TruncatedNormal,
                                    bias_initializer=Constant,
                                    name=name3)
        self.bna = layers.BatchNormalization(axis=bn_axis, gamma_initializer=RandomNormal,
                                             beta_initializer=Constant,
                                             name='d_bn_' + name2[3])
        self.bnb = layers.BatchNormalization(axis=bn_axis, gamma_initializer=RandomNormal,
                                             beta_initializer=Constant,
                                             name='d_bn' + name3[3])
        self.lrelu = layers.LeakyReLU(alpha=0.3)

    def call(self, inputs):
        x = self.conv2a(inputs)
        x = self.bna(x)
        x = self.lrelu(x)
        x = self.conv2b(x)
        x = self.bnb(x)
        x = self.lrelu(x)
        return x


class Discriminator(tf.keras.Model):
    """
    DCGAN discriminator
    """

    def __init__(self, data_format):

        super(Discriminator, self).__init__(name='')
        self.num_classes = 1001
        self.n_kernels = 300
        self.dim_per_kernel = 50
        self.batch_size = 2 * FLAGS.batch_size
        self.df_dim = FLAGS.df_dim
        self.image_size = FLAGS.image_size

        if data_format == 'channels_first':
            self._input_shape = [-1, 3, self.image_size, self.image_size]
            self.noise = tf.random_normal(
                [self.batch_size, 3, self.image_size, self.image_size])
            bn_axis = 1
        else:
            assert data_format == 'channels_last'
            self._input_shape = [-1, self.image_size, self.image_size, 3]
            self.noise = tf.random_normal(
                [self.batch_size, self.image_size, self.image_size, 3])
            bn_axis = -1

        def conv_block(filters, kernel_size, strides=(2, 2), padding='SAME', data_format='channels_first', name=''):
            return _Conv2d_block(
                filters,
                kernel_size,
                strides,
                padding,
                data_format,
                name)

        def minibatch_block(n_kernels, dim_per_kernel, name=''):
            return _Minibatch_block(
                n_kernels,
                dim_per_kernel,
                name)

        self.conv1 = layers.Conv2D(self.df_dim, 3, strides=2, padding='SAME',
                                   data_format=data_format,
                                   kernel_initializer=TruncatedNormal,
                                   bias_initializer=Constant,
                                   name='d_conv_h0')
        self.conv2 = conv_block([self.df_dim*2, self.df_dim*4], kernel_size=3,
                                strides=(2, 2), padding='SAME',
                                data_format=data_format,
                                name=['conv_block_1', 'd_h1_conv', 'd_h2_conv'])
        self.conv3 = conv_block([self.df_dim*4, self.df_dim*4], kernel_size=3,
                                strides=(1, 1), padding='SAME',
                                data_format=data_format,
                                name=['conv_block_2', 'd_h3_conv', 'd_h4_conv'])
        self.conv4 = conv_block([self.df_dim*8, self.df_dim*8], kernel_size=3,
                                strides=(2, 2), padding='SAME',
                                data_format=data_format,
                                name=['conv_block_3', 'd_h5_conv', 'd_h6_conv'])
        self.bna = layers.BatchNormalization(axis=bn_axis,
                                             gamma_initializer=RandomNormal,
                                             beta_initializer=Constant,
                                             name='d_bn_0')
        self.bnb = layers.BatchNormalization(axis=bn_axis,
                                             gamma_initializer=RandomNormal,
                                             beta_initializer=Constant,
                                             name='d_bn_7')
        self.linear1 = layers.Dense(self.df_dim*40,
                                    kernel_initializer=RandomNormal,
                                    bias_initializer=Constant,
                                    name='d_h7')
        self.linear2 = layers.Dense(self.num_classes,
                                    kernel_initializer=RandomNormal,
                                    bias_initializer=Constant,
                                    name='d_indiv_logits')
        self.minibatch = minibatch_block(self.n_kernels, self.dim_per_kernel,
                                         name='d_h')
        self.lrelu = layers.LeakyReLU(alpha=0.2)

    def call(self, inputs):
        x = tf.reshape(inputs, self._input_shape)
        x = x + self.noise
        x = self.conv1(x)
        x = self.bna(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = tf.reshape(x, [self.batch_size, -1])
        x = self.linear1(x)
        x = self.bnb(x)
        x = self.lrelu(x)
        x = self.minibatch(x)
        x = self.linear2(x)
        return x
