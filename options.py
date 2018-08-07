import os
import numpy
import tensorflow as tf
import pprint


def option_initialize():
    flag = tf.app.flags
    flag.DEFINE_boolean('train', True, 'True for training, False for test')
    flag.DEFINE_integer('iters',100000, 'Epoch to train')
    flag.DEFINE_float('lr', 0.001, 'learning rate for adam')
    flag.DEFINE_float('beta_a', 0.0, 'beta 1 for adam')
    flag.DEFINE_float('beta_b', 0.9, 'beta 2 for adam')
    flag.DEFINE_float('m', 1, 'm for vkl')
    flag.DEFINE_string('mode', 'VKL', 'which mode to train, VKL or gradient_penalty')
    flag.DEFINE_integer('CRTIC_ITERS', 5, 'number of update of G inside each epoch')
    flag.DEFINE_integer('lamda', 1, 'for gradient_penalty only')
    flag.DEFINE_integer('input_size', 158, 'input image size')
    flag.DEFINE_integer('output_size', 128, 'output image size')
    flag.DEFINE_integer('batch_size', 64, 'the size of image batch')
    flag.DEFINE_string('dataset', 'celebA', 'the name of dataset[celebA ,cifar]')
    flag.DEFINE_string('input_pattern', '*jpg', 'Glob pattern of filename of input image')
    flag.DEFINE_string('checkpoint_dir', 'checkpoints', 'dictory name to save checkpoints')
    flag.DEFINE_string('sample_dir', 'samples', 'dictory name to save samples')
    flag.DEFINE_boolean('crop', True, 'True for crop, False for not')
    flag.DEFINE_integer('z_dimension', 128, 'the dimensionality of z')
    flag.DEFINE_integer('sample_num',50000,'number of images to be sampled')


    options=flag.FLAGS

    if not os.path.exists(options.checkpoint_dir):
        os.mkdir(options.checkpoint_dir)

    if not os.path.exists(options.sample_dir):
        os.mkdir(options.sample_dir)

    pp=pprint.PrettyPrinter()
    pp.pprint(flag.FLAGS.__flags)
    print('options initialized successfully')

    return options




