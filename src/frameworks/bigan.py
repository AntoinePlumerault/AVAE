import tensorflow as tf
import numpy as np

from absl import logging, flags

from .models.dcbigan.encoder import Encoder
from .models.dcbigan.generator import Generator
from .models.dcbigan.discriminator import Discriminator

from .utils.losses import (kl_divergence, negative_log_likelyhood,
    generator_adversarial_loss_bigan, discriminator_adversarial_loss_bigan, 
    latent_loss)



FLAGS = flags.FLAGS

class BIGAN:
    def __init__(self, img_size, latent_dim, width_multiplier, learning_rate):
        self.latent_dim = latent_dim

        self.enc = Encoder(latent_dim, img_size, width_multiplier)
        self.gen = Generator(latent_dim, img_size, width_multiplier)
        self.dis = Discriminator(latent_dim, img_size, width_multiplier)

        self.enc.build(input_shape=(1, img_size[0], img_size[1], img_size[2]))
        self.gen.build(input_shape=(1, latent_dim))
        self.dis.build(input_shape=[
            (1, img_size[0], img_size[1], img_size[2]), 
            (1, latent_dim)])
        
        self.models = {
            'enc': self.enc,
            'gen': self.gen,
            'dis': self.dis,}
         
        self.optimizers = {
            'enc_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5),
            'gen_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5),
            'dis_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5)}

    @tf.function
    def forward(self, x_real, training=tf.constant(False)):
        batch_size = x_real.shape.as_list()[0]

        # latent to image
        z_g = tf.random.normal([batch_size, self.latent_dim])
        x_g = self.gen(z_g, training=True)
        
        # image to latent 
        x_e = x_real
        z_e, _ = self.enc(x_real, training=True)
        
        # discriminator
        s_g = self.dis([x_g, z_g], training=True)
        s_e = self.dis([x_e, z_e], training=True)

        losses = {
            'gen': tf.reduce_mean(
                generator_adversarial_loss_bigan(s_e, s_g)),
            'dis': tf.reduce_mean(
                discriminator_adversarial_loss_bigan(s_e, s_g)),
        }
        
        images = tf.concat([x_e, x_g], axis=0)
        
        return losses, images
    
    @tf. function
    def train_step(self, x_real):
        with tf.GradientTape(persistent=True) as tape:
            losses, images = self.forward(x_real, training=tf.constant(True))
            model_losses = {
                'gen': losses['gen'],
                'enc': losses['gen'] * 10.0,
                'dis': losses['dis']}
            
        for key, model in self.models.items():
            gradients = tape.gradient(model_losses[key], 
                model.trainable_variables)
            self.optimizers[key+'_opt'].apply_gradients(
                zip(gradients, model.trainable_variables))
    
    def eval_step(self, x_real):
        losses, images = self.forward(x_real)
        template = '{:6.2f} {:6.2f}'
        logging_message = template.format(*losses.values())
        z = self.get_latent_code(x_real)
        x = self.decode(z)
        images = tf.concat([x, images], axis=0)
        return logging_message, images
    
    @tf.function
    def get_latent_code(self, x):
        z_real_mu, _ = self.enc(x, training=True)
        return z_real_mu

    @tf.function
    def decode(self, z):
        x = self.gen(z, training=True)
        return x
    
    @tf.function
    def generate(self, N):
        z = tf.random.normal([N, self.latent_dim])
        x = self.decode(z)
        return x

