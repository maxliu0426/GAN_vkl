# GAN_vkl
implementation of GAN-vkl

You'd better to train serveal times with multiple seeds of weight initialization to get the best results

## Prerequisites

- Python 3.3+
- Tensorflow
- SciPy
- pillow
- (Optional) Large-scale CelebFaces Dataset and ImageNet



## Usage
Please download corresponding dataset in advance.
we assume CIFAR dataset is located in ./cifar, CELEBA and Imagenet are located in ./data/celebA and ./data/imagenet

First, download the inception model

    $ python download.py --outfile inception_score.model
    
the hyperparameter is in options.py. 
you could  choose the architeture you would like to train in Networks.py

To train a model with downloaded dataset:

    $ python main.py
    
if you want to train the toy model:

    $python gan_toy.py
    
 ## Results
 
 results on toy model:
 comparison of vkl constraint, gradient penalty and spectral normalization
 ![result1](images/Fig.1.jpg)

inception scores under different seettings；
 ![result2](images/Fig.5.jpg)
 
 variation of inception score with iteration on ResNet Based CNN
 ![result3](images/Fig.7.jpg)
 
 comparison of state-of-the-arts GAN models
  ![result4](images/Table.1.jpg)
  
  Vartiation of cm with iteration:
  ![result5](images/Fig.6.jpg)
 
 
    

  

