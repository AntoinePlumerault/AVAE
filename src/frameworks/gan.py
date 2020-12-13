import tensorflow as tf
import numpy as np

from absl import logging

from .models.dcgan.generator import Generator
from .models.dcgan.discriminator import Discriminator

from .utils.losses import (
    generator_adversarial_loss, discriminator_adversarial_loss)



class GAN:
    def __init__(self, img_size, latent_dim, width_multiplier, learning_rate):
        self.latent_dim = latent_dim
       
        self.gen = Generator(latent_dim, img_size, width_multiplier)
        self.dis = Discriminator(img_size, width_multiplier)
        self.models = {
            'gen': self.gen,
            'dis': self.dis}
        
        self.gen.build(input_shape=(1, latent_dim))
        self.dis.build(input_shape=(1, img_size[0], img_size[1], img_size[2]))
        
        self.optimizers = {
            'gen_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5),
            'dis_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5)}

    @tf.function
    def forward(self, x_real, training=tf.constant(False)):
        batch_size = x_real.shape.as_list()[0]
        z_eval = np.random.randn(
            batch_size, self.latent_dim).astype(np.float32)
        
        if training:
            z = tf.random.normal([batch_size, self.latent_dim])
        else:
            z = z_eval
        
        x_fake = self.gen(z, training=True)
        d_real = self.dis(x_real, training=True)
        d_fake = self.dis(x_fake, training=True)
        
        losses = {
            'gen': tf.reduce_mean(
                generator_adversarial_loss(d_fake)),
            'dis': tf.reduce_mean(
                discriminator_adversarial_loss(d_real, d_fake))}
        
        images = tf.concat([x_real, x_fake], axis=0)

        return losses, images
    
    @tf. function
    def train_step(self, x_real):
        with tf.GradientTape(persistent=True) as tape:
            losses, *_ = self.forward(x_real, training=tf.constant(True))
            
            model_losses = {
                'gen': losses['gen'],
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
        return logging_message, images

    @tf.function
    def generate(self, N):
        z = tf.random.normal([N, self.latent_dim])
        x = self.gen(z, training=True)
        return x