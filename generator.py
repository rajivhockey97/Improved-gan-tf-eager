import tensorflow as tf

layers = tf.keras.layers
seed = 1234
TruncatedNormal = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
Constant = tf.keras.initializers.Constant(value=0.0)
TruncatedNormal_out = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.15, seed=seed)
RandomNormal = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

flags = tf.app.flags

FLAGS = flags.FLAGS


class Generator(tf.keras.Model):
    """
    DCGAN generator
    """

    def __init__(self, data_format):

        super(Generator, self).__init__(name='')
        self.batch_size = FLAGS.batch_size
        self.gf_dim = 64
        self.half = self.gf_dim // 2
        self.quarter = self.gf_dim // 4
        self.eighth = self.gf_dim // 8
        self.sixteenth = self.gf_dim // 16

        if data_format == 'channels_first':
            bn_axis = 1
        else:
            assert data_format == 'channels_last'
            bn_axis = -1

        def make_z(shape, minval=-1., maxval=1., name='', dtype=tf.float32):
            assert dtype is tf.float32
            return tf.random_uniform(shape, minval=minval, maxval=maxval,
                                     name=name, dtype=tf.float32)

        self.z0 = make_z([self.batch_size, 4, 4, self.gf_dim], name='h0z')
        self.z1 = make_z([self.batch_size, 8, 8, self.gf_dim], name='h1z')
        self.z2 = make_z([self.batch_size, 16, 16, self.half], name='h2z')
        self.z3 = make_z([self.batch_size, 32, 32, self.quarter], name='h3z')
        self.z4 = make_z([self.batch_size, 64, 64, self.eighth], name='h4z')
        self.z5 = make_z([self.batch_size, 128, 128,
                          self.sixteenth], name='h5z')
        self.deconv2d1 = layers.Conv2DTranspose(self.gf_dim*4, 2, strides=(2, 2),
                                                kernel_initializer=TruncatedNormal,
                                                bias_initializer=Constant,
                                                name='g_h1')
        self.deconv2d2 = layers.Conv2DTranspose(self.gf_dim*2, 2, strides=(2, 2),
                                                kernel_initializer=TruncatedNormal,
                                                bias_initializer=Constant,
                                                name='g_h2')
        self.deconv2d3 = layers.Conv2DTranspose(self.gf_dim*1, 2, strides=(2, 2),
                                                kernel_initializer=TruncatedNormal,
                                                bias_initializer=Constant,
                                                name='g_h3')
        self.deconv2d4 = layers.Conv2DTranspose(self.gf_dim*1, 2, strides=(2, 2),
                                                kernel_initializer=TruncatedNormal,
                                                bias_initializer=Constant,
                                                name='g_h4')
        self.deconv2d5 = layers.Conv2DTranspose(self.gf_dim*1, 2, strides=(2, 2),
                                                kernel_initializer=TruncatedNormal,
                                                bias_initializer=Constant,
                                                name='g_h5')
        self.deconv2d6 = layers.Conv2DTranspose(3, 2, strides=(1, 1), padding='SAME',
                                                kernel_initializer=TruncatedNormal_out,
                                                bias_initializer=Constant,
                                                name='g_h6')
        self.linear1 = layers.Dense(self.gf_dim*8*4*4, kernel_initializer=RandomNormal,
                                    bias_initializer=Constant, name='g_h0_lin')
        self.bn1 = layers.BatchNormalization(axis=bn_axis, gamma_initializer=RandomNormal,
                                             beta_initializer=Constant, name='g_bn1')
        self.bn2 = layers.BatchNormalization(axis=bn_axis, gamma_initializer=RandomNormal,
                                             beta_initializer=Constant, name='g_bn2')
        self.bn3 = layers.BatchNormalization(axis=bn_axis, gamma_initializer=RandomNormal,
                                             beta_initializer=Constant, name='g_bn3')
        self.bn4 = layers.BatchNormalization(axis=bn_axis, gamma_initializer=RandomNormal,
                                             beta_initializer=Constant, name='g_bn4')
        self.bn5 = layers.BatchNormalization(axis=bn_axis, gamma_initializer=RandomNormal,
                                             beta_initializer=Constant, name='g_bn5')
        self.bn6 = layers.BatchNormalization(axis=bn_axis, gamma_initializer=RandomNormal,
                                             beta_initializer=Constant, name='g_bn6')
        self.relu = layers.Activation('relu')
        self.tanh = layers.Activation('tanh')

    def call(self, inputs):
        zs = [inputs]
        x = self.linear1(inputs)
        x = tf.reshape(x, [-1, 4, 4, self.gf_dim*8])
        x = self.bn1(x)
        x = self.relu(x)
        zs.append(self.z0)

        x = tf.concat([x, self.z0], 3)
        x = self.deconv2d1(x)
        x = self.bn2(x)
        x = self.relu(x)
        zs.append(self.z1)

        x = tf.concat([x, self.z1], 3)
        x = self.deconv2d2(x)
        x = self.bn3(x)
        x = self.relu(x)
        zs.append(self.z2)

        x = tf.concat([x, self.z2], 3)
        x = self.deconv2d3(x)
        x = self.bn4(x)
        x = self.relu(x)
        zs.append(self.z3)

        x = tf.concat([x, self.z3], 3)
        x = self.deconv2d4(x)
        x = self.bn5(x)
        x = self.relu(x)
        zs.append(self.z4)

        x = tf.concat([x, self.z4], 3)
        x = self.deconv2d5(x)
        x = self.bn6(x)
        x = self.relu(x)
        zs.append(self.z5)

        x = tf.concat([x, self.z5], 3)
        x = self.deconv2d6(x)
        x = self.tanh(x)
        return x, zs
