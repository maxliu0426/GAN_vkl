# GAN_vkl
implementation of GAN-vkl

## Prerequisites

- Python 3.3+
- Tensorflow
- SciPy
- pillow
- (Optional) Large-scale CelebFaces Dataset



## Usage
Please download corresponding dataset in advance.
we assume CIFAR dataset is located in ./cifar, CELEBA and Imagenet are located in ./data/celebA and ./data/imagenet

First, download the inception model

    $ python download.py --outfile inception_score.model
    
the hyperparameter is in options.py. 
you could also choose the architeture in Networks.py

To train a model with downloaded dataset:

    $ python main.py
    
if you want to train the toy model:

    $python gan_toy.py
    
 ##Results
 
 results on toy model:
 comparison of vkl constraint, gradient penalty and spectral normalization
 ![result1](images/Fig.1.jpg)
 
 comparison of state-of-the-arts GAN models
  ![result2](images/Table.1.jpg)
  
  Verfication of 
 
 
    

  

