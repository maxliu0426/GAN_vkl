import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import os
import scipy.misc
import Networks
import math

import argparse
import chainer
from chainer import cuda
from chainer import datasets
from chainer import serializers
from inception_score import Inception
from inception_score import inception_score
import glob
import urllib
import gzip
import pickle as pickle


class GAN():
    def __init__(self,options,sess):
        self.opt=options
        self.sess=sess
        #initialize batch norm layer

        #initialize input
        self.inputs     =tf.placeholder(tf.float32,[options.batch_size,self.opt.output_size,self.opt.output_size,3],name='real_images')
        self.z          =tf.placeholder(tf.float32,[options.batch_size,options.z_dimension],name='z')
        self.mean=[]

        #choose which model to use
        Nets = Networks.Networks_Libs(options)

        self.generator, self.discriminator, self.sampler=Nets.Get_Networks()

        self.G = self.generator(self.z)
        self.samples=self.sampler(self.z)

        self.D_real = self.discriminator(self.inputs)
        self.D_fake = self.discriminator(self.G,reuse=True)

        #proposed convergence measure
        alpha = tf.random_uniform([self.opt.batch_size,1], minval=0, maxval=1)
        alpha = tf.reshape(alpha,[self.opt.batch_size,1,1,1])
        self.interval = alpha * self.G + (1 - alpha) * self.inputs
        self.interval_d= self.discriminator(self.interval, reuse=True)
        self.gradients = tf.gradients(self.interval_d, self.interval)[0]
        self.convergence_measure = 1 / tf.sqrt(tf.reduce_sum(tf.square(self.gradients)))
        self.convergence_measure_sum = tf.summary.scalar("convergence_measure", self.convergence_measure)

        # compute gradient_penalty
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), axis=3))
        self.gradient_penalty = tf.reduce_mean((self.slopes - 1) ** 2)


        #compute the loss
        self.d_loss_real = tf.reduce_mean(self.D_real)
        self.d_loss_fake = tf.reduce_mean(self.D_fake)
        
        self.g_loss = -self.d_loss_fake+tf.maximum(tf.square(self.d_loss_fake) - self.opt.m, 0)

        self.d_loss_real_sum = tf.summary.scalar('d_loss_real', self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake', self.d_loss_fake)

        if self.opt.mode == "VKL":
            self.d_loss = self.d_loss_fake - self.d_loss_real + \
                          self.opt.lamda *tf.maximum(tf.square(self.d_loss_real) -  self.opt.m, 0) + \
                          self.opt.lamda *tf.maximum(tf.square(self.d_loss_fake) -  self.opt.m, 0)

     
        if self.opt.mode == "gradient_penalty":
            self.d_loss = self.d_loss_real - self.d_loss_fake + self.opt.lamda * self.gradient_penalty


        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)


        t_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(t_vars,print_info=True)
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.saver = tf.train.Saver(max_to_keep=150)
       
        self.d_optim = tf.train.AdamOptimizer(self.opt.lr, beta1=self.opt.beta_a, beta2=self.opt.beta_b).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.opt.lr, beta1=self.opt.beta_a, beta2=self.opt.beta_b).minimize(self.g_loss, var_list=self.g_vars)

        # initialize all variable
        tf.global_variables_initializer().run()

        self.d_sum = tf.summary.merge([self.d_loss_real_sum, self.d_loss_sum])
        self.g_sum=tf.summary.merge([self.d_loss_fake_sum,self.g_loss_sum])

        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
        self.sample_z = np.random.normal(0, 1, size=(self.opt.batch_size, self.opt.z_dimension))
        self.counter=1
        self.start_time=time.time()
        print('initialize success')

        could_load, checkpont_counter=self.load(self.opt.checkpoint_dir)
        if could_load:
            self.counter=checkpont_counter
            print('load success')
        else:
            print('load failed')

    def optimization(self,batch_image,batch_z):
        if self.counter==1:
            self.save_image(batch_image, './{}/sample.png'.format(self.opt.sample_dir), 8)
        for item in range(self.opt.CRTIC_ITERS):
            _, summary=self.sess.run([self.d_optim,self.d_sum],feed_dict={ self.inputs:batch_image,self.z:batch_z})
            self.writer.add_summary(summary,self.counter)

        errD_fake=self.d_loss_fake.eval({ self.z:batch_z})
        errD_real=self.d_loss_real.eval({self.inputs:batch_image})
        errG = self.g_loss.eval({self.z: batch_z})
        errD = self.d_loss.eval({self.z: batch_z, self.inputs: batch_image})

        print( "conter: [%4d] time: %4.4f, d_loss_real: %.8f, d_loss_fake: %.8f,d_loss:%.8f,g_loss: %.8f" \
            % (self.counter, time.time() - self.start_time, errD_real, errD_fake, errD, errG))


        _, summary = self.sess.run([self.g_optim, self.g_sum], feed_dict={self.z: batch_z})
        self.writer.add_summary(summary, self.counter)

        #calcalate the convergence measure
        if np.mod(self.counter, 100) == 1:
            samples = self.sess.run(self.samples, feed_dict={ self.z: self.sample_z })

            self.save_image(samples, './{}/{:06d}.png'.format(self.opt.sample_dir, self.counter),8)

            convergence_measure, convergence_measure_summary=self.sess.run([self.convergence_measure,self.convergence_measure_sum],
                                                                           feed_dict={self.z: batch_z,self.inputs: batch_image})
            self.writer.add_summary(convergence_measure_summary, self.counter)

            print("convergence_measure:%.8f"%(convergence_measure))
        #save the model
        if np.mod(self.counter,500)==2:
            self.save(self.opt.checkpoint_dir,self.counter)
        # calculate the inception score
        if np.mod(self.counter,500)==2 :
           self.image_sampler()

        self.counter+=1

     # for sampling after the training
    def sampling(self):
        iters = int(math.ceil(float(self.opt.sample_num) / float(self.opt.batch_size)))
        for iter in range(iters):
            sample_z = np.random.normal(0, 1, size=(self.opt.batch_size, self.opt.z_dimension))
            samples = self.sess.run(self.samples, feed_dict={self.z: sample_z})
            images = np.minimum(np.maximum((samples + 1.) / 2., 0), 1)
            self.save_image(images,'./{}/sample_{:06d}.png'.format(self.opt.sample_dir, iter),8)

    def save_image(self,images,path,size):
        images=np.minimum(np.maximum((images+1.)/2.,0),1)
        h,w= images.shape[1],images.shape[2]
        img=np.zeros((h*size,w*size,3))
        for idx, image in enumerate(images):
            i=idx%size
            j=idx//size
            img[j*h:j*h+h,i*w:i*w+w,:]=image
        img=np.squeeze(img)
        return scipy.misc.imsave(path,img)


    def save(self,checkpoint_dir,step):
        model_name='GAN.model'
        checkpoint_dir=os.path.join(checkpoint_dir,self.opt.dataset)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir,model_name),global_step=step)

    def load(self,checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.opt.dataset)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def image_sampler(self):
        iters = int(math.ceil(float(self.opt.sample_num) / float(self.opt.batch_size)))
        self.ims=[]
        for iter in range(iters):
            sample_z = np.random.normal(0, 1, size=(64, self.opt.z_dimension))
            samples = self.sess.run(self.samples, feed_dict={self.z: sample_z})
            self.ims.append(samples)
        # Load trained model
        ims=np.reshape(self.ims,(-1,32,32,3))
        ims=ims[:self.opt.sample_num].transpose(0,3,1,2)
        ims=((ims+1.0)*127.5).astype(np.uint8)
        ims=ims.astype('f')
        model = Inception()
        serializers.load_hdf5('inception_score.model', model)
        cuda.get_device(0).use()
        model.to_gpu()
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            mean, std = inception_score(model, ims)
        self.mean.append(mean)
        print('Inception score mean: %4.4f /  %4.4f'%(mean,max(self.mean)))
        print('Inception score std:', std)
        with open('data.txt','a',encoding='ascii') as f:
            f.write(str(self.counter))
            f.write(':')
            f.write(str(mean))
            f.write(',  ')
            f.write(str(std))
            f.write('\n')
        return mean, std

