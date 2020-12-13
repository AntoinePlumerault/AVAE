import tensorflow as tf
import numpy as np

from absl import logging, flags

from .models.dcvaegan.encoder import Encoder
from .models.dcvaegan.decoder import Decoder
from .models.dcvaegan.discriminator import Discriminator

from .utils.losses import (kl_divergence, negative_log_likelyhood, 
    generator_adversarial_loss, discriminator_adversarial_loss)



FLAGS = flags.FLAGS

class VAEGAN:
    def __init__(self, img_size, latent_dim, width_multiplier, learning_rate):
        self.latent_dim = latent_dim
       
        self.enc = Encoder(latent_dim, img_size, width_multiplier)
        self.dec = Decoder(latent_dim, img_size, width_multiplier)
        self.dis = Discriminator(img_size, width_multiplier)
        self.models = {
            'enc': self.enc,
            'dec': self.dec,
            'dis': self.dis}
        
        self.enc.build(input_shape=(1, img_size[0], img_size[1], img_size[2]))
        self.dec.build(input_shape=(1, latent_dim))
        self.dis.build(input_shape=(1, img_size[0], img_size[1], img_size[2]))

        self.optimizers = {
            'enc_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5),
            'dec_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5),
            'dis_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5)}

    @tf.function
    def forward(self, x_real, training=tf.constant(False)):
        batch_size = x_real.shape.as_list()[0]
        eps = tf.random.normal([batch_size, self.latent_dim])

        z_mu, z_log_sigma = self.enc(x_real, training=True)
        z = z_mu + tf.exp(z_log_sigma) * eps
        x_fake = self.dec(z, training=True)
        d_real, r_real = self.dis(x_real, training=True)
        d_fake, r_fake = self.dis(x_fake, training=True)
        
        z_gen = tf.random.normal([batch_size, self.latent_dim])
        x_gen = self.dec(z_gen, training=True)
        d_gen, _ = self.dis(x_gen, training=True)

        losses = {
            'kl': tf.reduce_mean(
                kl_divergence(z_mu, z_log_sigma)),
            'll': tf.reduce_mean(
                negative_log_likelyhood(r_real+0.0, r_fake)),
            'dis': tf.reduce_mean(
                discriminator_adversarial_loss(d_real, d_fake) +
                tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.zeros_like(d_gen), d_gen)) ,
            'gen': tf.reduce_mean(
                generator_adversarial_loss(d_fake) + 
                generator_adversarial_loss(d_gen))}
        
        images = tf.concat([x_real, x_fake], axis=0) 
        
        return losses, images
    
    @tf. function
    def train_step(self, x_real):
        with tf.GradientTape(persistent=True) as tape:
            losses, *_ = self.forward(x_real, training=tf.constant(True))
            gamma = 1e-2 # Official implementation
            model_losses = {
                'enc': FLAGS.beta * losses['kl'] + losses['ll'],
                'dec': gamma * losses['ll'] + losses['gen'],
                'dis': losses['dis']}
            
        for key, model in self.models.items():
            gradients = tape.gradient(model_losses[key], 
                model.trainable_variables)
            self.optimizers[key+'_opt'].apply_gradients(
                zip(gradients, model.trainable_variables))
    
    def eval_step(self, x_real):
        losses, images = self.forward(x_real)
        template = '{:6.2f} {:6.2f} {:6.2f} {:6.2f}'
        logging_message = template.format(*losses.values())
        return logging_message, images

    @tf.function
    def generate(self, N):
        z = tf.random.normal([N, self.latent_dim])
        x = self.dec(z, training=True)
        return x