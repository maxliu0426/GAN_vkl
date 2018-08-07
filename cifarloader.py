import numpy as np
import glob
import os
import urllib
import gzip
import pickle as pickle

def unpickle(file):
    with open(file,'rb') as fo:
       dict = pickle.load(fo,encoding='bytes')
    return dict[b'data'], dict[b'labels']

def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)


    return images



class cifarDataLoader():
    def __init__(self,options):
        self.batch_size=options.batch_size
        self.z_dimension=options.z_dimension
        self.images=cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','data_batch_6'], self.batch_size, './cifar')
        self.lens=self.images.shape[0]

    def batch_iters(self):
        return self.lens//self.batch_size

    def Loaddata(self,idx):
        batch=self.images[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_image=batch.reshape(-1,3,32,32).transpose([0,2,3,1])
        batch_image=batch_image/127.5-1
        batch_z=np.random.normal(0,1,[self.batch_size, self.z_dimension]).astype(np.float32)
        return batch_image,batch_z
