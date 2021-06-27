#from __future__ import absolute_import, division, print_function

import os, sys
sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.save_images
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.cond_batchnorm

import tflib.ops.deconv2d
#import tflib.data_loader
import tflib.lsun_label
import tflib.ops.layernorm
import tflib.plot
import pdb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="LSUN_10", help = "dataset. anime or face. ")
parser.add_argument('--result_dir', type=str, default="result", help = "pretrained BigGAN model")
parser.add_argument('--source_domain', type=str, default="imagenet", help = "save frequency in iteration. currently no eval is implemented and just model saving and sample generation is performed" )
parser.add_argument('--iters', type=int, default=10000)
parser.add_argument('--critic_iters', type=int, default=5)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
args=parser.parse_args()


DATA_DIR = 'data/'+args.data_dir
RESULT_DIR = './'+args.result_dir
SAMPLES_DIR = RESULT_DIR + '/samples/'
MODEL_DIR = RESULT_DIR + '/model/'

SOURCE_DOMAIN =args.source_domain # imagenet, places, celebA, bedroom,
ACGAN = True
if ACGAN: 
    PRETRAINED_MODEL = './transfer_model/conditional/%s/wgan-gp.model'%SOURCE_DOMAIN 
else: 
    PRETRAINED_MODEL = './transfer_model/unconditional/%s/wgan-gp.model'%SOURCE_DOMAIN 

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
SAVE_SAMPLES_STEP = 50 # Generate and save samples every SAVE_SAMPLES_STEP
CHECKPOINT_STEP = 4000

ITER_START = 0


ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 1. # How to scale generator's ACGAN loss relative to WGAN loss

N_CLASSES = args.n_classes
DIM = 64 # Model dimensionality
N_PIXELS = 64

# Settings for TTUR and orig
CRITIC_ITERS = args.critic_iters # How many iterations to train the critic for
D_LR = 0.00001
G_LR = 0.00001
BETA1_D = 0.0
BETA1_G = 0.0
#FID_STEP = 333 # FID evaluation every FID_STEP
FID_STEP = 250 # FID evaluation every FID_STEP
ITERS = args.iters # How many iterations to train for


# Switch on and off batchnormalizaton for the discriminator
# and the generator. Default is on for both.
BN_D=True
BN_G=True

# Log subdirectories are automatically created from
# the above settings and the current timestamp.
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = args.batch_size # Batch size. Must be a multiple of N_GPUS
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = N_PIXELS * N_PIXELS * 3 # Number of pixels in each iamge


# Create directories if necessary
if not os.path.exists(SAMPLES_DIR):
  print("*** create sample dir %s" % SAMPLES_DIR)
  os.makedirs(SAMPLES_DIR)
if not os.path.exists(MODEL_DIR):
  print("*** create checkpoint dir %s" % MODEL_DIR)
  os.makedirs(MODEL_DIR)

#lib.print_model_settings(locals().copy(), LOG_DIR)
lib.print_model_settings(locals().copy())

def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """

    # For actually generating decent samples, use this one
    return GoodGenerator, GoodDiscriminator


DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Normalize(name, axes, inputs, labels = None):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:        
        if labels is not None and ACGAN:
            return lib.ops.cond_batchnorm.Batchnorm(name,axes,inputs,labels=labels,n_labels=N_CLASSES)
        else:
            return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)
        
def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(a=output, perm=[0,2,3,1])
    output = tf.compat.v1.depth_to_space(input=output, block_size=2)
    output = tf.transpose(a=output, perm=[0,3,1,2])
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(a=output, perm=[0,2,3,1])
    output = tf.compat.v1.depth_to_space(input=output, block_size=2)
    output = tf.transpose(a=output, perm=[0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True, bn=False, labels = None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    if bn:
      output = Normalize(name+'.BN1', [0,2,3], output, labels = labels)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    if bn:
      output = Normalize(name+'.BN2', [0,2,3], output, labels = labels)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


# ! Generators

def GoodGenerator(n_samples, noise=None, dim=DIM, nonlinearity=tf.nn.relu, bn=BN_G, labels = None):
    if noise is None:
        noise = tf.random.normal([n_samples, 128])

    ## supports 32x32 images
    fact = N_PIXELS // 16
    output = lib.ops.linear.Linear('Generator.Input', 128, fact*fact*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, fact, fact])
    output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up', bn=bn, labels = labels)
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up', bn=bn, labels = labels)
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up', bn=bn, labels = labels)
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up', bn=bn, labels = labels)
    if bn:
      output = Normalize('Generator.OutputN', [0,2,3], output, labels = labels)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
    output = tf.tanh(output)
    
    return tf.reshape(output, [-1, OUTPUT_DIM])
# ! Discriminators

def GoodDiscriminator(inputs, dim=DIM, bn=BN_D):
    fact = N_PIXELS // 16

    output = tf.reshape(inputs, [-1, 3, N_PIXELS, N_PIXELS])
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, resample='down', bn=bn)

    output = tf.reshape(output, [-1, fact*fact*8*dim])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', fact*fact*8*dim, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', fact*fact*8*dim, N_CLASSES, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None
Generator, Discriminator = GeneratorAndDiscriminator()

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as session:

    all_real_data_conv = tf.compat.v1.placeholder(tf.int32, shape=[BATCH_SIZE, 3, N_PIXELS, N_PIXELS])
    all_real_labels = tf.compat.v1.placeholder(tf.int32, shape=[BATCH_SIZE])
    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)
    #fake_data_splits = []
    #for i, device in enumerate(DEVICES):
    #    with tf.device(device):
    #        fake_data_splits.append(Generator(BATCH_SIZE/len(DEVICES), labels = labels_splits[i]))
   
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(all_real_data_conv,axis=0, num_or_size_splits=len(DEVICES), )
        
    print("batch size HHHHaaaaa:",split_real_data_conv)
    gen_costs, disc_costs, disc_real_acgan_costs, disc_fake_acgan_costs = [],[],[],[]
    disc_acgan_real_accs, disc_acgan_fake_accs = [], []
    for device_index, (device, real_data_conv, real_labels) in enumerate(zip(DEVICES, split_real_data_conv, labels_splits)):
    
        with tf.device(device):
            
            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE//len(DEVICES), OUTPUT_DIM],)
            fake_labels = tf.cast(tf.random.uniform([BATCH_SIZE//len(DEVICES)])*10, tf.int32)            
            fake_data = Generator(BATCH_SIZE//len(DEVICES), bn=BN_G, labels = fake_labels)

            disc_real, disc_real_acgan = Discriminator(real_data)
            disc_fake, disc_fake_acgan = Discriminator(fake_data)

            gen_cost = -tf.reduce_mean(input_tensor=disc_fake)
            disc_wgan = tf.reduce_mean(input_tensor=disc_fake) - tf.reduce_mean(input_tensor=disc_real)

            alpha = tf.random.uniform(shape=[BATCH_SIZE//len(DEVICES),1], minval=0., maxval=1. )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            #pdb.set_trace()
            gradients = tf.gradients(ys=Discriminator(interpolates, bn=BN_D)[0], xs=interpolates)[0]
            slopes = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(gradients), axis=[1]))
            gradient_penalty = tf.reduce_mean(input_tensor=(slopes-1.)**2)
            disc_wgan_pure = disc_wgan
            disc_wgan += LAMBDA*gradient_penalty
            disc_cost = disc_wgan

            if ACGAN:                
                disc_real_acgan_costs.append(tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_real_acgan, labels=real_labels)))
                disc_fake_acgan_costs.append(tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)))                
                disc_cost += ACGAN_SCALE * tf.add_n(disc_real_acgan_costs)
                gen_cost += ACGAN_SCALE_G * tf.add_n(disc_fake_acgan_costs)
                disc_acgan_real_accs.append(tf.reduce_mean(
                    input_tensor=tf.cast(tf.equal(tf.cast(tf.argmax(input=disc_real_acgan, axis=1), dtype=tf.int32), real_labels ), tf.float32)))
                disc_acgan_fake_accs.append(tf.reduce_mean(
                    input_tensor=tf.cast(tf.equal(tf.cast(tf.argmax(input=disc_fake_acgan, axis=1), dtype=tf.int32), fake_labels ), tf.float32)))

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)    
    if ACGAN:
        disc_acgan_real_acc = tf.add_n(disc_acgan_real_accs) / len(DEVICES)
        disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES)    

    gen_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=G_LR, beta1=BETA1_G, beta2=0.9).minimize(gen_cost,
                                      var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=D_LR, beta1=BETA1_D, beta2=0.9).minimize(disc_cost,
                                       var_list=lib.params_with_name('Discriminator.'))
    # For generating samples
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    fixed_noise_samples = Generator(100, labels = fixed_labels, noise=fixed_noise)

    def generate_image(frame):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((100, 3, N_PIXELS, N_PIXELS)), "/content/image.png")


    fake_labels_100 = tf.cast(tf.random.uniform([100])*N_CLASSES, tf.int32)
    samples_100 = Generator(100, labels = fake_labels_100)

    # Dataset iterator
    #train_gen, dev_gen = lib.data_loader.load(BATCH_SIZE, DATA_DIR, DATASET)
    train_gen, dev_gen = lib.lsun_label.load(BATCH_SIZE, DATA_DIR, NUM_TRAIN = 10000)

    def inf_train_gen():
        while True:
            for (images,labels) in train_gen():
                yield images,labels
    gen = inf_train_gen()
    # Save a batch of ground-truth samples
    _x,_ =next( inf_train_gen())
    _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE//N_GPUS]})
    _x_r = ((_x_r+1.)*(255.99//2)).astype('int32')
    lib.save_images.save_images(_x_r.reshape((BATCH_SIZE//N_GPUS, 3, DIM, DIM)), '%s/samples_groundtruth.png' % SAMPLES_DIR)

    session.run(tf.compat.v1.global_variables_initializer())
    # just save and restore all parameters,instead of learning rate 
    ckpt_saver = tf.compat.v1.train.Saver(lib.params_with_name('Generator') + lib.params_with_name('Discriminator.'))
    ckpt_saver.restore(session, PRETRAINED_MODEL)
    ckpt_saver = tf.compat.v1.train.Saver()

    for it in range(ITERS):
        iteration = it + ITER_START
        start_time = time.time()

        # Train generator
        if iteration > 0:
            _gen_cost, _disc_fake_acgan_costs, _ = session.run([gen_cost, disc_fake_acgan_costs, gen_train_op])
            lib.plot.plot('%s/g-cost'%RESULT_DIR, _gen_cost)
            if ACGAN:
                lib.plot.plot('%s/acgan-fake'%RESULT_DIR, np.mean(_disc_fake_acgan_costs))

        disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _images, _labels = next(gen)
            if ACGAN:
                _disc_cost, _disc_wgan, _disc_wgan_pure, _disc_real_acgan_costs, _disc_acgan_real_acc, _disc_acgan_fake_acc, _ = session.run(
                    [disc_cost, disc_wgan, disc_wgan_pure, disc_real_acgan_costs, disc_acgan_real_acc, disc_acgan_fake_acc, disc_train_op], 
                        feed_dict={all_real_data_conv: _images, all_real_labels: _labels})
            else:
                _disc_cost, _disc_wgan, _disc_wgan_pure,  _ = session.run(
                    [disc_cost, disc_wgan, disc_wgan_pure, disc_train_op], 
                        feed_dict={all_real_data_conv: _images, all_real_labels: _labels})
            #_disc_cost = session.run(disc_cost, feed_dict={all_real_data_conv: _data})
        
        lib.plot.plot('%s/d-cost'%RESULT_DIR, _disc_cost)
        lib.plot.plot('%s/wgan-pure'%RESULT_DIR, _disc_wgan_pure)
        lib.plot.plot('%s/penalty'%RESULT_DIR, _disc_wgan - _disc_wgan_pure)
        if ACGAN:
            lib.plot.plot('%s/wgan'%RESULT_DIR, _disc_wgan)
            lib.plot.plot('%s/acgan-real'%RESULT_DIR, np.mean(_disc_real_acgan_costs))
            lib.plot.plot('%s/real_acc'%RESULT_DIR, _disc_acgan_real_acc)
            lib.plot.plot('%s/fake_acc'%RESULT_DIR, _disc_acgan_fake_acc)
        lib.plot.plot('%s/time'%RESULT_DIR, time.time() - start_time)

        generate_image(iteration)
        if iteration < 100 or iteration % SAVE_SAMPLES_STEP == 0:
            generate_image(iteration)
        # Save checkpoint
        if iteration % CHECKPOINT_STEP == 0:
            ckpt_saver.save(session, os.path.join(MODEL_DIR, "WGAN_GP.model"), iteration)
        
        lib.plot.flush(path = RESULT_DIR)
        lib.plot.tick()
