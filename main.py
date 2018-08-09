import tensorflow as tf
import model
import options
import datasetloader
import cifarloader
run_config=tf.ConfigProto()
run_config.gpu_options.allow_growth=True
# run optimization
with tf.Session(config=run_config) as sess:
    # load options
    opt = options.option_initialize()
    # load dataset
    if opt.dataseyt == cifar:
        data_loader = cifarloader.cifarDataLoader(opt)      #for image generation using cifar
    else:
        data_loader = datasetloader.datasetloader(opt)     #for image generation using celebA or imagenet
    
    Gan=model.GAN(opt,sess)
    batch_iters=data_loader.batch_iters()
    print('batch_iters:',batch_iters)

    if opt.train == False:
       Gan.sampling()
    else:
        epochs=(opt.iters-Gan.counter)//batch_iters
        for epoch in range(epochs):
            for idx in range(0, batch_iters):
                batch_images, batch_z = data_loader.Loaddata(idx)
                Gan.optimization(batch_images, batch_z)
        print(Gan.mean)
        print(max(Gan.mean))





