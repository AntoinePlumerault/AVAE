import os
import sys
import numpy as np
import tensorflow as tf
import random

from absl import app, flags, logging

from utils.save_images import save_images



# Allow memory growth and optimizations
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.optimizer.set_jit(True)

def main(argv):

    if FLAGS.framework == 'gan': from frameworks.gan import GAN as Framework
    if FLAGS.framework == 'vae': from frameworks.vae import VAE as Framework
    if FLAGS.framework == 'avae': from frameworks.avae import AVAE as Framework
    if FLAGS.framework == 'bigan': from frameworks.bigan import BIGAN as Framework
    if FLAGS.framework == 'vaegan': from frameworks.vaegan import VAEGAN as Framework

    if FLAGS.dataset == 'svhn': from datasets import svhn as dataset
    if FLAGS.dataset == 'celeba': from datasets import celeba as dataset
    if FLAGS.dataset == 'bedroom': from datasets import bedroom as dataset
    if FLAGS.dataset == 'cifar10': from datasets import cifar10 as dataset
    if FLAGS.dataset == 'cifar100': from datasets import cifar100 as dataset
    if FLAGS.dataset == 'dsprites': from datasets import dsprites as dataset
    
    # Create working directories
    experiment_dir  = os.path.join(FLAGS.output_dir, 
        FLAGS.experiment_name, FLAGS.framework, FLAGS.dataset)
    
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    saved_model_dir = os.path.join(experiment_dir, 'saved_models')
    images_dir = os.path.join(experiment_dir, 'images')
    test_dir = os.path.join(experiment_dir, 'test')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    if FLAGS.mode == 'train':
        logging.get_absl_handler().use_absl_log_file('logs', experiment_dir)

    # Load dataset
    dataset, img_size = dataset.load(FLAGS.batch_size, 
        mode='test' if FLAGS.mode == 'test_mse_lpips' else 'train')

    # Load framework
    framework = Framework(
        img_size=img_size, 
        latent_dim=FLAGS.latent_dim, 
        width_multiplier=FLAGS.width_multiplier, 
        learning_rate=tf.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=FLAGS.initial_learning_rate, 
            decay_steps=50000, decay_rate=FLAGS.decay_rate))

    # Manage checkpoints
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(0), **framework.models, **framework.optimizers)
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(experiment_dir, 'checkpoints'), max_to_keep=1)
    
    # Restore checkpoint
    if FLAGS.restore: ckpt.restore(manager.latest_checkpoint)

# ================================ TESTING =====================================

    if 'test' in FLAGS.mode:
        # load models
        for name, model in framework.models.items():
            model.load_weights(os.path.join(
                saved_model_dir, '{}_{:06d}.h5'.format(name, 50001)))
        
        # --------------------------- LPIPS & MSE ------------------------------
        
        if FLAGS.mode == 'test_mse_lpips':
            if FLAGS.framework in ['avae', 'vae', 'bigan', 'vaegan']:
                from utils.lpips import lpips
        
                N = 1000
                N_batch = (N+FLAGS.batch_size-1)//FLAGS.batch_size
                
                lpips_value = 0.0
                mse_value = 0.0
                for i, features in dataset.enumerate():
                    i = i.numpy()
                    _, images = framework.eval_step(features['image'])
                    if FLAGS.framework == 'avae':
                        x_real, _, x_fake = tf.split(images, 3)
                    if FLAGS.framework == 'bigan':
                        x_fake, x_real, _ = tf.split(images, 3)
                    if FLAGS.framework == 'vae':
                        x_real, x_fake = tf.split(images, 2)
                    if FLAGS.framework == 'vaegan':
                        x_real, x_fake = tf.split(images, 2)
                    lpips_value += 1.0/(i+1.0) * (
                        tf.reduce_mean(lpips.lpips(x_real, x_fake)).numpy() - lpips_value)
                    mse_value += 1.0/(i+1.0) * (
                        tf.reduce_mean(tf.math.square(x_fake - x_real)).numpy() - mse_value)
            
            else: lpips_value, mse_value = -1, -1
            fid_value = -1
        # ------------------------------- FID ----------------------------------
        
        if FLAGS.mode == 'test_fid':
            from utils.frechet_inception_distance import FrechetInceptionDistance

            N = 50000
            N_batch = (N+FLAGS.batch_size-1)//FLAGS.batch_size
            fid = FrechetInceptionDistance(dataset, N)
            gen_images = []
            i=0
            for _ in range(N_batch):
                i+=1
                print(i)
                images = framework.generate(FLAGS.batch_size)
                gen_images.append(images)
            fid_value = fid(gen_images) 
            mse_value, lpips_value = -1, -1
            
        # ---------------------------- Write results ---------------------------

        output_file = os.path.join(test_dir, FLAGS.mode+'.csv')
        with open(output_file, 'w') as f:
            f.write('mse,lpips,fid\n')
            f.write('{:<5.2f},{:<5.2f},{:<5.2f}'.format(
                mse_value, lpips_value, fid_value))

        
# ================================ TRAINING ====================================

    if FLAGS.mode == 'train':
        for step, features in dataset.enumerate(FLAGS.initial_step):
            framework.train_step(features['image'])

            if step % FLAGS.eval_freq == 0:
                logging_message, images = framework.eval_step(features['image'])
                save_images(np.array(images), os.path.join(
                    images_dir, 'image_{}.png'.format(step)))
                logging.info('step: {:06d} - '.format(step) + logging_message)

            if step % FLAGS.save_freq == 0 and step != 0:
                manager.save()  
            
            ckpt.step.assign_add(1)
            if step == FLAGS.final_step + 1: break
        
        # Save model
        for name, model in framework.models.items():
            model.save_weights(os.path.join(
                saved_model_dir, '{}_{:06d}.h5'.format(name, step)))

if __name__ == '__main__':

    FLAGS = flags.FLAGS

    flags.DEFINE_string('data_dir', os.path.join('..', 'data'), "datasets directory")
    flags.DEFINE_string('output_dir', os.path.join('..', 'outputs'), "outputs directory")
    flags.DEFINE_string('experiment_name', 'test', "")
    flags.DEFINE_enum('mode', 'train', ['train', 'test_fid', 'test_mse_lpips'], "")

    flags.DEFINE_string('dataset', 'celeba', "")
    flags.DEFINE_enum('framework', 'vae', ['vae', 'gan', 'bigan', 'avae', 'vaegan'], "")
    flags.DEFINE_integer('width_multiplier', 64, "")
    flags.DEFINE_integer('latent_dim', 100, "")

    flags.DEFINE_integer('initial_step', 0, "")
    flags.DEFINE_integer('final_step', 50000, "")
    flags.DEFINE_integer('save_freq', 10000, "")
    flags.DEFINE_integer('eval_freq', 1000, "")

    flags.DEFINE_bool('restore', False, "")
    flags.DEFINE_integer('batch_size', 64, "")
    flags.DEFINE_float('initial_learning_rate', 0.0002, "")
    flags.DEFINE_float('decay_rate', 0.02, "")
    flags.DEFINE_float('sigma', 1.0, "")
    flags.DEFINE_float('beta', 1.0, "")

    app.run(main)