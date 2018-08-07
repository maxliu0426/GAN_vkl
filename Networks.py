import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import os
import scipy.misc

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
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


def conv2d(input_, output_dim, k_size=4, d_size=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_size, k_size, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_size, d_size, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape, k_size=4, d_size=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_size, k_size, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides=[1, d_size, d_size, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],initializer=tf.constant_initializer(bias_start))

    return tf.matmul(input_, matrix) + bias

def nonlinear_g(input):
    return tf.nn.relu(input)

def nonlinear_d(input):
    return tf.nn.leaky_relu(input)



class Networks_Libs():
    def __init__(self,opt):
        self.opt=opt
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')
        self.g_bn6 = batch_norm(name='g_bn6')
        self.g_bn7 = batch_norm(name='g_bn7')
        self.g_bn8 = batch_norm(name='g_bn8')
        self.g_bn9 = batch_norm(name='g_bn9')
        self.g_bn10 = batch_norm(name='g_bn10')
        self.g_bn11 = batch_norm(name='g_bn11')


    def Get_Networks(self):
        return self.SCNN_celebA_g, self.SCNN_celebA_d, self.SCNN_celebA_s


    #testing area for 32*32
    #  apply skip connection in doubled SCNN_d
    def d_t(self, image, reuse=False):
        with tf.variable_scope('d_a',reuse=tf.AUTO_REUSE) as scope:
            if reuse:
                scope.reuse_variables()
            #compute the image output
            h0 = tf.nn.leaky_relu(conv2d(image, 64, k_size=3, d_size=1, name='d_h0_conv'))
            h0 = tf.nn.leaky_relu(conv2d(h0, 64, k_size=3, d_size=1, name='d_h1_conv'))+h0
            h1 = tf.nn.avg_pool(h0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p0')
            h2 = tf.nn.leaky_relu((conv2d(h1, 128, k_size=3, d_size=1, name='d_h2_conv')))
            h2 = tf.nn.leaky_relu((conv2d(h2, 128, k_size=3, d_size=1, name='d_h3_conv')))+h2
            h3 = tf.nn.avg_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p1')
            h4 = tf.nn.leaky_relu(conv2d(h3, 256, k_size=3, d_size=1, name='d_h4_conv'))
            h4 = tf.nn.leaky_relu(conv2d(h4, 256, k_size=3, d_size=1, name='d_h5_conv'))+h4
            h5 = tf.nn.avg_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p3')
            h8 = linear(tf.reshape(h5, [self.opt.batch_size, -1]), 1, 'd_h8_linear')

            return h8

    def g_t(self, z):
        with tf.variable_scope('g_a') as scope:
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))
            h0 = tf.nn.leaky_relu(self.g_bn1(conv2d(h0,512,k_size=3,d_size=1,name='g_h0_conv')))+h0
            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn2(conv2d(h1, 256, k_size=3, d_size=1, name='g_h1_conv')))
            h2 = tf.nn.leaky_relu(self.g_bn3(conv2d(h2, 256, k_size=3, d_size=1, name='g_h2_conv')))+h2
            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn4(conv2d(h3, 128, k_size=3, d_size=1, name='g_h3_conv')))
            h4 = tf.nn.leaky_relu(self.g_bn5(conv2d(h4, 128, k_size=3, d_size=1, name='g_h4_conv')))+h4
            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn6(conv2d(h5, 64, k_size=3, d_size=1, name='g_h5_conv')))
            h6 = tf.nn.leaky_relu(self.g_bn7(conv2d(h6, 64, k_size=3, d_size=1, name='g_h6_conv')))+h6
            h7 = conv2d(h6, 3, k_size=3, d_size=1, name='g_h9_conv')
            return tf.nn.tanh(h7)

    def s_t(self, z):
        with tf.variable_scope('g_a') as scope:
            scope.reuse_variables()
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))
            h0 = tf.nn.leaky_relu(self.g_bn1(conv2d(h0, 512, k_size=3, d_size=1, name='g_h0_conv'))) + h0
            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn2(conv2d(h1, 256, k_size=3, d_size=1, name='g_h1_conv')))
            h2 = tf.nn.leaky_relu(self.g_bn3(conv2d(h2, 256, k_size=3, d_size=1, name='g_h2_conv'))) + h2
            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn4(conv2d(h3, 128, k_size=3, d_size=1, name='g_h3_conv')))
            h4 = tf.nn.leaky_relu(self.g_bn5(conv2d(h4, 128, k_size=3, d_size=1, name='g_h4_conv'))) + h4
            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn6(conv2d(h5, 64, k_size=3, d_size=1, name='g_h5_conv')))
            h6 = tf.nn.leaky_relu(self.g_bn7(conv2d(h6, 64, k_size=3, d_size=1, name='g_h6_conv'))) + h6
            h7 = conv2d(h6, 3, k_size=3, d_size=1, name='g_h9_conv')
            return tf.nn.tanh(h7)


    # DenseNet for 64*64

    def DenseNetBlock(self,input,dim_out, BN_use= False, length=3, name='g_'):
        h1 = input
        h = input
        for len in range(length):
            if BN_use== True:
                h = conv2d(nonlinear_d(tf.contrib.layers.batch_norm(h1,
                      decay=0.9,
                      updates_collections=None,
                      epsilon=1e-5,
                      scale=True,
                      is_training=True,
                      scope=name+'bn'+str(len))), dim_out, k_size=3, d_size=1, name=name +'conv_'+ str(len))
            else:
                h = conv2d(nonlinear_d(h1), dim_out, k_size=3, d_size=1, name=name + 'conv_'+str(len))
            h1 = tf.concat([h,h1],axis=3)

        return h

    def DenseNet_g(self,z):
        with tf.variable_scope('generator') as scope:
            z_ = linear(z, 32 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 32])

            h = self.DenseNetBlock(h, 32, True, name='g_1_')
            h = conv2d(self.g_bn0(h), 32, k_size=1, d_size=1, name='g_conv_1')
            h = tf.keras.layers.UpSampling2D(size=(2, 2))(h)

            h = self.DenseNetBlock(h, 32, True, name='g_2_')
            h = conv2d(self.g_bn1(h), 32, k_size=1, d_size=1, name='g_conv_2')
            h = tf.keras.layers.UpSampling2D(size=(2, 2))(h)

            h = self.DenseNetBlock(h, 32, True, name='g_3_')
            h = conv2d(self.g_bn2(h), 32, k_size=1, d_size=1, name='g_conv_3')
            h = tf.keras.layers.UpSampling2D(size=(2, 2))(h)

            h = self.DenseNetBlock(h, 32, True, name='g_4_')
            h = conv2d(self.g_bn3(h), 32, k_size=1, d_size=1, name='g_conv_4')
            h = tf.keras.layers.UpSampling2D(size=(2, 2))(h)

            h = self.DenseNetBlock(h, 32, True, name='g_5_')
            h = conv2d(tf.nn.relu(self.g_bn4(h)), 3, k_size=3, d_size=1, name='g_conv_5')
            print(h.shape)
            return tf.nn.tanh(h)

    def DenseNet_s(self, z):
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            z_ = linear(z, 32 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 32])

            h = self.DenseNetBlock(h, 32, True, name='g_1_')
            h = conv2d(self.g_bn0(h), 32, k_size=1, d_size=1, name='g_conv_1')
            h = tf.keras.layers.UpSampling2D(size=(2, 2))(h)

            h = self.DenseNetBlock(h, 32, True, name='g_2_')
            h = conv2d(self.g_bn1(h), 32, k_size=1, d_size=1, name='g_conv_2')
            h = tf.keras.layers.UpSampling2D(size=(2, 2))(h)

            h = self.DenseNetBlock(h, 32, True, name='g_3_')
            h = conv2d(self.g_bn2(h), 32, k_size=1, d_size=1, name='g_conv_3')
            h = tf.keras.layers.UpSampling2D(size=(2, 2))(h)

            h = self.DenseNetBlock(h, 32, True, name='g_4_')
            h = conv2d(self.g_bn3(h), 32, k_size=1, d_size=1, name='g_conv_4')
            h = tf.keras.layers.UpSampling2D(size=(2, 2))(h)

            h = self.DenseNetBlock(h, 32, True, name='g_5_')
            h = conv2d(tf.nn.relu(self.g_bn4(h)), 3, k_size=3, d_size=1, name='g_conv_5')
            return tf.nn.tanh(h)

    def DenseNet_d(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            h = conv2d(image, 32, k_size=3,d_size=1, name='d_conv_1')

            h = self.DenseNetBlock(h,32,False,name='d_1_')
            h = tf.nn.avg_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p0')

            h = self.DenseNetBlock(h, 32, False, name='d_2_')
            h = tf.nn.avg_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p1')

            h = self.DenseNetBlock(h, 32, False, name='d_3_')
            h = tf.nn.avg_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p2')

            h = self.DenseNetBlock(h, 32, False, name='d_4_')
            h = tf.nn.avg_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p3')

            h = self.DenseNetBlock(h, 32, False, name='d_5_')
            print(h.shape)
            h = linear(tf.reshape(h, [self.opt.batch_size, -1]), 1, 'd_h4_linear')
            return h

    # standard CNN for cifar------------------------------------------------------------------
    def SCNN_d(self, image, reuse=False):
        with tf.variable_scope('DCGAN_discriminator_nb') as scope:
            if reuse:
                scope.reuse_variables()
            h0 = tf.nn.leaky_relu(conv2d(image, 64, k_size=3, d_size=1, name='d_h0_conv'))
            h1 = tf.nn.leaky_relu((conv2d(h0, 128, name='d_h1_conv')))
            h2 = tf.nn.leaky_relu(conv2d(h1, 256, name='d_h2_conv'))
            h3 = tf.nn.leaky_relu(conv2d(h2, 512, name='d_h3_conv'))
            h4 = linear(tf.reshape(h3, [self.opt.batch_size, -1]), 1, 'd_h4_linear')

            return h4

    def SCNN_g(self, z):
        with tf.variable_scope('DCGAN_generator') as scope:
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.relu(self.g_bn0(h))

            h1 = deconv2d(h0, [self.opt.batch_size, 8, 8, 256], name='g_h1')
            h2 = tf.nn.relu(self.g_bn1(h1))

            h3 = deconv2d(h2, [self.opt.batch_size, 16, 16, 128], name='g_h2')
            h4 = tf.nn.relu(self.g_bn2(h3))

            h5 = deconv2d(h4, [self.opt.batch_size, 32, 32, 64], name='g_h3')
            h6 = tf.nn.relu(self.g_bn3(h5))
            h7 = conv2d(h6, 3, k_size=3, d_size=1, name='g_h4')
            # h7 = deconv2d(h6, [self.opt.batch_size, 64, 64, 3], name='g_h4')

            return tf.nn.tanh(h7)

    def SCNN_s(self, z):
        with tf.variable_scope('DCGAN_generator') as scope:
            scope.reuse_variables()
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.relu(self.g_bn0(h, train=False))

            h1 = deconv2d(h0, [self.opt.batch_size, 8, 8, 256], name='g_h1')
            h2 = tf.nn.relu(self.g_bn1(h1, train=False))

            h3 = deconv2d(h2, [self.opt.batch_size, 16, 16, 128], name='g_h2')
            h4 = tf.nn.relu(self.g_bn2(h3, train=False))

            h5 = deconv2d(h4, [self.opt.batch_size, 32, 32, 64], name='g_h3')
            h6 = tf.nn.relu(self.g_bn3(h5, train=False))
            h7 = conv2d(h6, 3, k_size=3, d_size=1, name='g_h4')
            # h7 = deconv2d(h6, [self.opt.batch_size, 64, 64, 3], name='g_h4')

            return tf.nn.tanh(h7)

    # standard CNN doubled for cifar
    def SCNN_double_d(self, image, reuse=False):
        with tf.variable_scope('d_a', reuse=tf.AUTO_REUSE) as scope:
            if reuse:
                scope.reuse_variables()
            # compute the image output
            h0 = tf.nn.leaky_relu(conv2d(image, 64, k_size=3, d_size=1, name='d_h0_conv'))
            h0 = tf.nn.leaky_relu(conv2d(h0, 64, k_size=3, d_size=1, name='d_h1_conv')) + h0
            h1 = tf.nn.avg_pool(h0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p0')
            h2 = tf.nn.leaky_relu((conv2d(h1, 128, k_size=3, d_size=1, name='d_h2_conv')))
            h2 = tf.nn.leaky_relu((conv2d(h2, 128, k_size=3, d_size=1, name='d_h3_conv'))) + h2
            h3 = tf.nn.avg_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p1')
            h4 = tf.nn.leaky_relu(conv2d(h3, 256, k_size=3, d_size=1, name='d_h4_conv'))
            h4 = tf.nn.leaky_relu(conv2d(h4, 256, k_size=3, d_size=1, name='d_h5_conv')) + h4
            h5 = tf.nn.avg_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p3')
            h6 = tf.nn.leaky_relu(conv2d(h5, 512, k_size=3, d_size=1, name='d_h6_conv'))
            h6 = tf.nn.leaky_relu(conv2d(h6, 512, k_size=3, d_size=1, name='d_h7_conv')) + h6
            h8 = linear(tf.reshape(h6, [self.opt.batch_size, -1]), 1, 'd_h8_linear')
            return h8

    def SCNN_double_g(self, z):
        with tf.variable_scope('g_a') as scope:
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))
            h0 = tf.nn.leaky_relu(self.g_bn1(conv2d(h0, 512, k_size=3, d_size=1, name='g_h0_conv'))) + h0
            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn2(conv2d(h1, 256, k_size=3, d_size=1, name='g_h1_conv')))
            h2 = tf.nn.leaky_relu(self.g_bn3(conv2d(h2, 256, k_size=3, d_size=1, name='g_h2_conv'))) + h2
            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn4(conv2d(h3, 128, k_size=3, d_size=1, name='g_h3_conv')))
            h4 = tf.nn.leaky_relu(self.g_bn5(conv2d(h4, 128, k_size=3, d_size=1, name='g_h4_conv'))) + h4
            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn6(conv2d(h5, 64, k_size=3, d_size=1, name='g_h5_conv')))
            h6 = tf.nn.leaky_relu(self.g_bn7(conv2d(h6, 64, k_size=3, d_size=1, name='g_h6_conv'))) + h6
            h9 = conv2d(h6, 3, k_size=3, d_size=1, name='g_h9_conv')
            return tf.nn.tanh(h9)

    def SCNN_double_s(self, z):
        with tf.variable_scope('g_a') as scope:
            scope.reuse_variables()
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))
            h0 = tf.nn.leaky_relu(self.g_bn1(conv2d(h0, 512, k_size=3, d_size=1, name='g_h0_conv'))) + h0
            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn2(conv2d(h1, 256, k_size=3, d_size=1, name='g_h1_conv')))
            h2 = tf.nn.leaky_relu(self.g_bn3(conv2d(h2, 256, k_size=3, d_size=1, name='g_h2_conv'))) + h2
            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn4(conv2d(h3, 128, k_size=3, d_size=1, name='g_h3_conv')))
            h4 = tf.nn.leaky_relu(self.g_bn5(conv2d(h4, 128, k_size=3, d_size=1, name='g_h4_conv'))) + h4
            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn6(conv2d(h5, 64, k_size=3, d_size=1, name='g_h5_conv')))
            h6 = tf.nn.leaky_relu(self.g_bn7(conv2d(h6, 64, k_size=3, d_size=1, name='g_h6_conv'))) + h6
            h9 = conv2d(h6, 3, k_size=3, d_size=1, name='g_h9_conv')
            return tf.nn.tanh(h9)
    #------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    # ResNet Based CNN for cifar
    def RCNN_g(self, z):
        with tf.variable_scope('WGAN_GP_generator') as scope:
            z_ = linear(z, 128 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 128])
            # residual block 1
            h0 = tf.nn.relu(self.g_bn0(h))
            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = conv2d(h1, 128, k_size=3, d_size=1, name='g_h1')
            h3 = tf.nn.relu(self.g_bn1(h2))
            h4 = conv2d(h3, 128, k_size=3, d_size=1, name='g_h2')
            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h)
            h6 = conv2d(h5, 128, k_size=1, d_size=1, name='g_h3')
            h7 = h4 + h6

            # residul block 2
            h8 = tf.nn.relu(self.g_bn2(h7))
            h9 = tf.keras.layers.UpSampling2D(size=(2, 2))(h8)
            h10 = conv2d(h9, 128, k_size=3, d_size=1, name='g_h4')
            h11 = tf.nn.relu(self.g_bn3(h10))
            h12 = conv2d(h11, 128, k_size=3, d_size=1, name='g_h5')
            h13 = tf.keras.layers.UpSampling2D(size=(2, 2))(h7)
            h14 = conv2d(h13, 128, k_size=1, d_size=1, name='g_h6')
            h15 = h12 + h14

            # residual block 3
            h16 = tf.nn.relu(self.g_bn4(h15))
            h17 = tf.keras.layers.UpSampling2D(size=(2, 2))(h16)
            h18 = conv2d(h17, 128, k_size=3, d_size=1, name='g_h7')
            h19 = tf.nn.relu(self.g_bn5(h18))
            h20 = conv2d(h19, 128, k_size=3, d_size=1, name='g_h8')
            h21 = tf.keras.layers.UpSampling2D(size=(2, 2))(h15)
            h22 = conv2d(h21, 128, k_size=1, d_size=1, name='g_h9')
            h23 = h20 + h22

            h24 = tf.nn.relu(self.g_bn6(h23))
            h25 = conv2d(h24, 3, k_size=3, d_size=1, name='g_h10')

            return tf.nn.tanh(h25)

    def RCNN_s(self, z):
        with tf.variable_scope('WGAN_GP_generator') as scope:
            scope.reuse_variables()
            z_ = linear(z, 128 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 128])
            # residual block 1
            h0 = tf.nn.relu(self.g_bn0(h))
            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = conv2d(h1, 128, k_size=3, d_size=1, name='g_h1')
            h3 = tf.nn.relu(self.g_bn1(h2))
            h4 = conv2d(h3, 128, k_size=3, d_size=1, name='g_h2')
            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h)
            h6 = conv2d(h5, 128, k_size=1, d_size=1, name='g_h3')
            h7 = h4 + h6

            # residul block 2
            h8 = tf.nn.relu(self.g_bn2(h7))
            h9 = tf.keras.layers.UpSampling2D(size=(2, 2))(h8)
            h10 = conv2d(h9, 128, k_size=3, d_size=1, name='g_h4')
            h11 = tf.nn.relu(self.g_bn3(h10))
            h12 = conv2d(h11, 128, k_size=3, d_size=1, name='g_h5')
            h13 = tf.keras.layers.UpSampling2D(size=(2, 2))(h7)
            h14 = conv2d(h13, 128, k_size=1, d_size=1, name='g_h6')
            h15 = h12 + h14

            # residual block 3
            h16 = tf.nn.relu(self.g_bn4(h15))
            h17 = tf.keras.layers.UpSampling2D(size=(2, 2))(h16)
            h18 = conv2d(h17, 128, k_size=3, d_size=1, name='g_h7')
            h19 = tf.nn.relu(self.g_bn5(h18))
            h20 = conv2d(h19, 128, k_size=3, d_size=1, name='g_h8')
            h21 = tf.keras.layers.UpSampling2D(size=(2, 2))(h15)
            h22 = conv2d(h21, 128, k_size=1, d_size=1, name='g_h9')
            h23 = h20 + h22

            h24 = tf.nn.relu(self.g_bn6(h23))
            h25 = conv2d(h24, 3, k_size=3, d_size=1, name='g_h10')

            return tf.nn.tanh(h25)

    def RCNN_d(self, image, reuse=False):
        with tf.variable_scope('WGAN_GP_discriminator_nb') as scope:
            if reuse:
                scope.reuse_variables()
            # preprocessing
            h0 = conv2d(image, 128, k_size=3, d_size=1, name='d_h0')

            # resiudual block 1
            h1 = tf.nn.relu(h0)
            h2 = conv2d(h1, 128, k_size=3, d_size=1, name='d_h1')
            h3 = tf.nn.relu(h2)
            h4 = conv2d(h3, 128, k_size=3, d_size=1, name='d_h2')
            h5 = tf.nn.avg_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p0')
            h6 = tf.nn.avg_pool(h0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p1')
            h7 = conv2d(h6, 128, k_size=1, d_size=1, name='d_h3')
            h8 = h7 + h5

            # residual block2
            h9 = tf.nn.relu(h8)
            h10 = conv2d(h9, 128, k_size=3, d_size=1, name='d_h4')
            h11 = tf.nn.relu(h10)
            h12 = conv2d(h11, 128, k_size=3, d_size=1, name='d_h5')
            h13 = tf.nn.avg_pool(h12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p2')
            h14 = tf.nn.avg_pool(h8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p3')
            h15 = conv2d(h14, 128, k_size=1, d_size=1, name='d_h6')
            h16 = h15 + h13

            # residual block 3
            h17 = tf.nn.relu(h16)
            h18 = conv2d(h17, 128, k_size=3, d_size=1, name='d_h7')
            h19 = tf.nn.relu(h18)
            h20 = conv2d(h19, 128, k_size=3, d_size=1, name='d_h8')
            h21 = tf.nn.avg_pool(h20, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p4')
            h22 = tf.nn.avg_pool(h16, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p5')
            h23 = conv2d(h22, 128, k_size=1, d_size=1, name='d_h9')
            h24 = h23 + h21

            h25 = linear(tf.reshape(h24, [self.opt.batch_size, -1]), 1, 'd_h13_linear')

            return h25
        # --------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------
    # image generation on 64*64 resolution

    def da(self, image, reuse=False):
        with tf.variable_scope('d_a',reuse=tf.AUTO_REUSE) as scope:
            if reuse:
                scope.reuse_variables()
            #compute the image output
            h0 = tf.nn.leaky_relu(conv2d(image, 64, k_size=3, d_size=1, name='d_h0_conv'))
            h0 = tf.nn.leaky_relu(conv2d(h0, 64, k_size=3, d_size=1, name='d_h1_conv'))+h0
            h1 = tf.nn.avg_pool(h0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p0')
            h2 = tf.nn.leaky_relu((conv2d(h1, 128, k_size=3, d_size=1, name='d_h2_conv')))
            h2 = tf.nn.leaky_relu((conv2d(h2, 128, k_size=3, d_size=1, name='d_h3_conv')))+h2
            h3 = tf.nn.avg_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p1')
            h4 = tf.nn.leaky_relu(conv2d(h3, 256, k_size=3, d_size=1, name='d_h4_conv'))
            h4 = tf.nn.leaky_relu(conv2d(h4, 256, k_size=3, d_size=1, name='d_h5_conv'))+h4
            h5 = tf.nn.avg_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p3')
            h6 = tf.nn.leaky_relu(conv2d(h5, 512, k_size=3, d_size=1, name='d_h6_conv'))
            h6 = tf.nn.leaky_relu(conv2d(h6, 512, k_size=3, d_size=1, name='d_h7_conv'))+h6
            h7 = tf.nn.avg_pool(h6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p4')
            h8 = linear(tf.reshape(h7, [self.opt.batch_size, -1]), 1, 'd_h8_linear')

            return h8

    def ga(self, z):
        with tf.variable_scope('g_a') as scope:
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))
            h0 = tf.nn.leaky_relu(self.g_bn1(conv2d(h0,512,k_size=3,d_size=1,name='g_h0_conv')))+h0
            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn2(conv2d(h1, 256, k_size=3, d_size=1, name='g_h1_conv')))
            h2 = tf.nn.leaky_relu(self.g_bn3(conv2d(h2, 256, k_size=3, d_size=1, name='g_h2_conv')))+h2
            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn4(conv2d(h3, 128, k_size=3, d_size=1, name='g_h3_conv')))
            h4 = tf.nn.leaky_relu(self.g_bn5(conv2d(h4, 128, k_size=3, d_size=1, name='g_h4_conv')))+h4
            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn6(conv2d(h5, 64, k_size=3, d_size=1, name='g_h5_conv')))
            h6 = tf.nn.leaky_relu(self.g_bn7(conv2d(h6, 64, k_size=3, d_size=1, name='g_h6_conv')))+h6
            h7 = tf.keras.layers.UpSampling2D(size=(2, 2))(h6)
            h8 = tf.nn.leaky_relu(self.g_bn8(conv2d(h7, 32, k_size=3, d_size=1, name='g_h7_conv')))
            h8 = tf.nn.leaky_relu(self.g_bn9(conv2d(h8, 32, k_size=3, d_size=1, name='g_h8_conv')))+h8
            h9 = conv2d(h8, 3, k_size=3, d_size=1, name='g_h9_conv')

            return tf.nn.tanh(h9)

    def sa(self, z):
        with tf.variable_scope('g_a') as scope:
            scope.reuse_variables()
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))
            h0 = tf.nn.leaky_relu(self.g_bn1(conv2d(h0, 512, k_size=3, d_size=1, name='g_h0_conv'))) + h0
            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn2(conv2d(h1, 256, k_size=3, d_size=1, name='g_h1_conv')))
            h2 = tf.nn.leaky_relu(self.g_bn3(conv2d(h2, 256, k_size=3, d_size=1, name='g_h2_conv'))) + h2
            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn4(conv2d(h3, 128, k_size=3, d_size=1, name='g_h3_conv')))
            h4 = tf.nn.leaky_relu(self.g_bn5(conv2d(h4, 128, k_size=3, d_size=1, name='g_h4_conv'))) + h4
            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn6(conv2d(h5, 64, k_size=3, d_size=1, name='g_h5_conv')))
            h6 = tf.nn.leaky_relu(self.g_bn7(conv2d(h6, 64, k_size=3, d_size=1, name='g_h6_conv'))) + h6
            h7 = tf.keras.layers.UpSampling2D(size=(2, 2))(h6)
            h8 = tf.nn.leaky_relu(self.g_bn8(conv2d(h7, 32, k_size=3, d_size=1, name='g_h7_conv')))
            h8 = tf.nn.leaky_relu(self.g_bn9(conv2d(h8, 32, k_size=3, d_size=1, name='g_h8_conv'))) + h8
            h9 = conv2d(h8, 3, k_size=3, d_size=1, name='g_h9_conv')

            return tf.nn.tanh(h9)
    #----------------------------------------------------------------------------------------------
    # image generation on 128*128 resolution
    def SCNN_celebA_d(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            h0 = tf.nn.leaky_relu(conv2d(image, 64, k_size=3, d_size=1, name='d_h0_conv'))
            h1 = tf.nn.avg_pool(h0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p0')
            h2 = tf.nn.leaky_relu((conv2d(h1, 128, k_size=3, d_size=1, name='d_h1_conv')))
            h3 = tf.nn.avg_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p1')
            h4 = tf.nn.leaky_relu(conv2d(h3, 256, k_size=3, d_size=1, name='d_h2_conv'))
            h5 = tf.nn.avg_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p2')
            h6 = tf.nn.leaky_relu(conv2d(h5, 512, k_size=3, d_size=1, name='d_h3_conv'))
            h7 = tf.nn.avg_pool(h6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p3')
            h8 = tf.nn.leaky_relu(conv2d(h7, 512, k_size=3, d_size=1, name='d_h4_conv'))
            h9 = tf.nn.avg_pool(h8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p4')
            h10 = linear(tf.reshape(h9, [self.opt.batch_size, -1]), 1, 'd_h5_linear')

            return h10

    def SCNN_celebA_g(self, z):
        with tf.variable_scope('generator') as scope:
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))

            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn1(conv2d(h1, 256, k_size=3, d_size=1, name='g_h0_conv')))

            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn2(conv2d(h3, 128, k_size=3, d_size=1, name='g_h1_conv')))

            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn3(conv2d(h5, 64, k_size=3, d_size=1, name='g_h2_conv')))

            h7 = tf.keras.layers.UpSampling2D(size=(2, 2))(h6)
            h8 = tf.nn.leaky_relu(self.g_bn4(conv2d(h7, 32, k_size=3, d_size=1, name='g_h3_conv')))

            h9 = tf.keras.layers.UpSampling2D(size=(2, 2))(h8)
            h10 = tf.nn.leaky_relu(self.g_bn5(conv2d(h9, 32, k_size=3, d_size=1, name='g_h4_conv')))

            h11 = conv2d(h10, 3, k_size=3, d_size=1, name='g_h5_conv')

            return tf.nn.tanh(h11)

    def SCNN_celebA_s(self, z):
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))

            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn1(conv2d(h1, 256, k_size=3, d_size=1, name='g_h0_conv')))

            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn2(conv2d(h3, 128, k_size=3, d_size=1, name='g_h1_conv')))

            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn3(conv2d(h5, 64, k_size=3, d_size=1, name='g_h2_conv')))

            h7 = tf.keras.layers.UpSampling2D(size=(2, 2))(h6)
            h8 = tf.nn.leaky_relu(self.g_bn4(conv2d(h7, 32, k_size=3, d_size=1, name='g_h3_conv')))

            h9 = tf.keras.layers.UpSampling2D(size=(2, 2))(h8)
            h10 = tf.nn.leaky_relu(self.g_bn5(conv2d(h9, 32, k_size=3, d_size=1, name='g_h4_conv')))

            h11 = conv2d(h10, 3, k_size=3, d_size=1, name='g_h5_conv')

            return tf.nn.tanh(h11)


    def SCNN_celebA_double_d(self, image, reuse=False):
        with tf.variable_scope('d_a', reuse=tf.AUTO_REUSE) as scope:
            if reuse:
                scope.reuse_variables()
            # compute the image output
            h0 = tf.nn.leaky_relu(conv2d(image, 32, k_size=3, d_size=1, name='d_h0_conv'))
            h0 = tf.nn.leaky_relu(conv2d(h0, 32, k_size=3, d_size=1, name='d_h1_conv')) + h0
            h1 = tf.nn.avg_pool(h0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p0')
            h2 = tf.nn.leaky_relu((conv2d(h1, 64, k_size=3, d_size=1, name='d_h2_conv')))
            h2 = tf.nn.leaky_relu((conv2d(h2, 64, k_size=3, d_size=1, name='d_h3_conv'))) + h2
            h3 = tf.nn.avg_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p1')
            h4 = tf.nn.leaky_relu(conv2d(h3, 128, k_size=3, d_size=1, name='d_h4_conv'))
            h4 = tf.nn.leaky_relu(conv2d(h4, 128, k_size=3, d_size=1, name='d_h5_conv')) + h4
            h5 = tf.nn.avg_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p3')
            h6 = tf.nn.leaky_relu(conv2d(h5, 256, k_size=3, d_size=1, name='d_h6_conv'))
            h6 = tf.nn.leaky_relu(conv2d(h6, 256, k_size=3, d_size=1, name='d_h7_conv')) + h6
            h7 = tf.nn.avg_pool(h6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p4')
            h8 = tf.nn.leaky_relu(conv2d(h7, 512, k_size=3, d_size=1, name='d_h8_conv'))
            h8 = tf.nn.leaky_relu(conv2d(h8, 512, k_size=3, d_size=1, name='d_h9_conv')) + h8
            h9 = tf.nn.avg_pool(h8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p5')
            h10 = linear(tf.reshape(h9, [self.opt.batch_size, -1]), 1, 'd_h10_linear')

            return h10

    def SCNN_celebA_double_g(self, z):
        with tf.variable_scope('g_a') as scope:
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))
            h0 = tf.nn.leaky_relu(self.g_bn1(conv2d(h0, 512, k_size=3, d_size=1, name='g_h0_conv'))) + h0
            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn2(conv2d(h1, 256, k_size=3, d_size=1, name='g_h1_conv')))
            h2 = tf.nn.leaky_relu(self.g_bn3(conv2d(h2, 256, k_size=3, d_size=1, name='g_h2_conv'))) + h2
            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn4(conv2d(h3, 128, k_size=3, d_size=1, name='g_h3_conv')))
            h4 = tf.nn.leaky_relu(self.g_bn5(conv2d(h4, 128, k_size=3, d_size=1, name='g_h4_conv'))) + h4
            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn6(conv2d(h5, 64, k_size=3, d_size=1, name='g_h5_conv')))
            h6 = tf.nn.leaky_relu(self.g_bn7(conv2d(h6, 64, k_size=3, d_size=1, name='g_h6_conv'))) + h6
            h7 = tf.keras.layers.UpSampling2D(size=(2, 2))(h6)
            h8 = tf.nn.leaky_relu(self.g_bn8(conv2d(h7, 32, k_size=3, d_size=1, name='g_h7_conv')))
            h8 = tf.nn.leaky_relu(self.g_bn9(conv2d(h8, 32, k_size=3, d_size=1, name='g_h8_conv'))) + h8

            h9 = tf.keras.layers.UpSampling2D(size=(2, 2))(h8)
            h10 = tf.nn.leaky_relu(self.g_bn10(conv2d(h9, 16, k_size=3, d_size=1, name='g_h9_conv')))
            h10 = tf.nn.leaky_relu(self.g_bn11(conv2d(h10, 16, k_size=3, d_size=1, name='g_h10_conv'))) + h10
            h11 = conv2d(h10, 3, k_size=3, d_size=1, name='g_h11_conv')

            return tf.nn.tanh(h11)

    def SCNN_celebA_double_s(self, z):
        with tf.variable_scope('g_a') as scope:
            scope.reuse_variables()
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))
            h0 = tf.nn.leaky_relu(self.g_bn1(conv2d(h0, 512, k_size=3, d_size=1, name='g_h0_conv'))) + h0
            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn2(conv2d(h1, 256, k_size=3, d_size=1, name='g_h1_conv')))
            h2 = tf.nn.leaky_relu(self.g_bn3(conv2d(h2, 256, k_size=3, d_size=1, name='g_h2_conv'))) + h2
            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn4(conv2d(h3, 128, k_size=3, d_size=1, name='g_h3_conv')))
            h4 = tf.nn.leaky_relu(self.g_bn5(conv2d(h4, 128, k_size=3, d_size=1, name='g_h4_conv'))) + h4
            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn6(conv2d(h5, 64, k_size=3, d_size=1, name='g_h5_conv')))
            h6 = tf.nn.leaky_relu(self.g_bn7(conv2d(h6, 64, k_size=3, d_size=1, name='g_h6_conv'))) + h6
            h7 = tf.keras.layers.UpSampling2D(size=(2, 2))(h6)
            h8 = tf.nn.leaky_relu(self.g_bn8(conv2d(h7, 32, k_size=3, d_size=1, name='g_h7_conv')))
            h8 = tf.nn.leaky_relu(self.g_bn9(conv2d(h8, 32, k_size=3, d_size=1, name='g_h8_conv'))) + h8

            h9 = tf.keras.layers.UpSampling2D(size=(2, 2))(h8)
            h10 = tf.nn.leaky_relu(self.g_bn10(conv2d(h9, 16, k_size=3, d_size=1, name='g_h9_conv')))
            h10 = tf.nn.leaky_relu(self.g_bn11(conv2d(h10, 16, k_size=3, d_size=1, name='g_h10_conv'))) + h10
            h11 = conv2d(h10, 3, k_size=3, d_size=1, name='g_h11_conv')

            return tf.nn.tanh(h11)



    #base model for all expriments . it's for 3*64*64
    def discriminator(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            h0 = tf.nn.leaky_relu(conv2d(image, 64, k_size=3, d_size=1, name='d_h0_conv'))
            h1 = tf.nn.avg_pool(h0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p0')
            h2 = tf.nn.leaky_relu((conv2d(h1, 128, k_size=3, d_size=1, name='d_h1_conv')))
            h3 = tf.nn.avg_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p1')
            h4 = tf.nn.leaky_relu(conv2d(h3, 256, k_size=3, d_size=1, name='d_h2_conv'))
            h5 = tf.nn.avg_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p3')
            h6 = tf.nn.leaky_relu(conv2d(h5, 512, k_size=3, d_size=1, name='d_h3_conv'))
            h7 = tf.nn.avg_pool(h6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='d_p4')
            h8 = linear(tf.reshape(h7, [self.opt.batch_size, -1]), 1, 'd_h4_linear')

            return h8

    def generator(self, z):
        with tf.variable_scope('generator') as scope:
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))

            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn1(conv2d(h1, 256, k_size=3, d_size=1, name='g_h0_conv')))

            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn2(conv2d(h3, 128, k_size=3, d_size=1, name='g_h1_conv')))

            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn3(conv2d(h5, 64, k_size=3, d_size=1, name='g_h2_conv')))

            h7 = tf.keras.layers.UpSampling2D(size=(2, 2))(h6)
            h8 = tf.nn.leaky_relu(self.g_bn4(conv2d(h7, 32, k_size=3, d_size=1, name='g_h3_conv')))

            h9 = conv2d(h8, 3, k_size=3, d_size=1, name='g_h4_conv')

            return tf.nn.tanh(h9)

    def Sampler(self, z):
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            z_ = linear(z, 512 * 4 * 4, 'g_h0_linear')
            h = tf.reshape(z_, [-1, 4, 4, 512])
            h0 = tf.nn.leaky_relu(self.g_bn0(h))

            h1 = tf.keras.layers.UpSampling2D(size=(2, 2))(h0)
            h2 = tf.nn.leaky_relu(self.g_bn1(conv2d(h1, 256, k_size=3, d_size=1, name='g_h0_conv')))

            h3 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)
            h4 = tf.nn.leaky_relu(self.g_bn2(conv2d(h3, 128, k_size=3, d_size=1, name='g_h1_conv')))

            h5 = tf.keras.layers.UpSampling2D(size=(2, 2))(h4)
            h6 = tf.nn.leaky_relu(self.g_bn3(conv2d(h5, 64, k_size=3, d_size=1, name='g_h2_conv')))

            h7 = tf.keras.layers.UpSampling2D(size=(2, 2))(h6)
            h8 = tf.nn.leaky_relu(self.g_bn4(conv2d(h7, 32, k_size=3, d_size=1, name='g_h3_conv')))

            h9 = conv2d(h8, 3, k_size=3, d_size=1, name='g_h4_conv')

            return tf.nn.tanh(h9)










