import os
from glob import glob
import scipy.misc
import numpy as np

def transform(image,input_size,resize_size,crop=True):
    if crop:
        cropped_image=center_crop(image,input_size,resize_size)
    else:
        cropped_image=scipy.misc.imresize(image,[resize_size,resize_size])

    return np.array(cropped_image)/127.5-1

def center_crop(image,input_size,resize_size):
    h,w=image.shape[:2]
    j=int(round((h-input_size)/2.))
    i=int(round((w-input_size)/2.))
    return scipy.misc.imresize(image[j:j+input_size,i:i+input_size],[resize_size,resize_size])

def get_image(path,input_size,resize_size,crop=True):
    image=scipy.misc.imread(path).astype(np.float)
    return transform(image,input_size,resize_size,crop)

class datasetloader():
    def __init__(self,options):
        self.name=options.dataset
        self.input_patern=options.input_pattern
        self.data_paths=glob(os.path.join('./data',options.dataset,options.input_pattern))
        self.batch_size=options.batch_size
        self.input_size=options.input_size
        self.resize_szie=options.output_size
        self.crop=options.crop
        self.z_dimension=options.z_dimension

    def batch_iters(self):
        return len(self.data_paths)//self.batch_size

    def Loaddata(self,idx):
        batch_paths=self.data_paths[idx*self.batch_size:(idx+1)*self.batch_size]
        batch=[get_image(batch_path, self.input_size, self.resize_szie, self.crop) for batch_path in batch_paths]
        batch_images=np.array(batch).astype(np.float32)
        batch_z=np.random.normal(0,1,[self.batch_size, self.z_dimension]).astype(np.float32)
        return batch_images,batch_z




