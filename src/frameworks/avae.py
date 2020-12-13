import tensorflow as tf
import numpy as np

from absl import logging, flags

from .models.dcvae.encoder import Encoder
from .models.dcvae.decoder import Decoder
from .models.dcgan.generator import Generator
from .models.dcgan.discriminator import Discriminator

from .utils.losses import (kl_divergence, negative_log_likelyhood,
    generator_adversarial_loss, discriminator_adversarial_loss, latent_loss)



FLAGS = flags.FLAGS

class AVAE:
    def __init__(self, img_size, latent_dim, width_multiplier, learning_rate):
        self.latent_dim = latent_dim

        self.gen = Generator(latent_dim, img_size, width_multiplier)
        self.dis = Discriminator(img_size, width_multiplier)
        self.enc = Encoder(latent_dim, img_size, width_multiplier)
        self.dec = Decoder(latent_dim, img_size, width_multiplier)

        self.gen.build(input_shape=(1, 2*latent_dim))
        self.dis.build(input_shape=(1, img_size[0], img_size[1], img_size[2]))
        self.enc.build(input_shape=(1, img_size[0], img_size[1], img_size[2]))
        self.dec.build(input_shape=(1, latent_dim))
        
        self.models = {
            'enc': self.enc,
            'dec': self.dec,
            'gen': self.gen,
            'dis': self.dis,}
         
        self.optimizers = {
            'enc_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5),
            'dec_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5),
            'gen_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5),
            'dis_opt': tf.optimizers.Adam(learning_rate, beta_1=0.5)}

    @tf.function
    def forward(self, x_real, training=tf.constant(False)):
        batch_size = x_real.shape.as_list()[0]
        eps = tf.random.normal([batch_size, self.latent_dim])
        xi = tf.random.normal([batch_size, self.latent_dim])
        z_sample = np.random.randn(batch_size, self.latent_dim).astype(np.float32)

        z_real_mu, z_real_log_sigma = self.enc(x_real, training=True)
        z_real = z_real_mu + tf.exp(z_real_log_sigma) * eps
        x_real_mu = self.dec(z_real, training=True)

        if training: z_fake = tf.random.normal([batch_size, self.latent_dim])
        else: z_fake = z_real

        # latent vae 
        x_fake = self.gen(tf.concat([z_fake, xi], axis=1), training=True)
        d_real = self.dis(x_real, training=True)
        d_fake = self.dis(x_fake, training=True)
        
        z_fake_mu, z_fake_log_sigma = self.enc(x_fake, training=True)

        losses = {
            'kl': tf.reduce_mean(
                kl_divergence(z_real_mu, z_real_log_sigma)),
            'll': tf.reduce_mean(
                negative_log_likelyhood(x_real+0.0, x_real_mu)),
            'gen': tf.reduce_mean(
                generator_adversarial_loss(d_fake)),
            'dis': tf.reduce_mean(
                discriminator_adversarial_loss(d_real, d_fake)),
            'lat': tf.reduce_mean(
                latent_loss(z_fake, z_fake_mu, z_fake_log_sigma))}
        
        images = tf.concat([x_real, x_real_mu, x_fake], axis=0)
        
        return losses, images
    
    @tf. function
    def train_step(self, x_real):
        with tf.GradientTape(persistent=True) as tape:
            losses, images = self.forward(x_real, training=tf.constant(True))
            
            model_losses = {
                'enc': FLAGS.beta * losses['kl'] + losses['ll'],
                'dec': losses['ll'],
                'gen': losses['gen'] + losses['lat'],
                'dis': losses['dis']}
            
        for key, model in self.models.items():
            gradients = tape.gradient(model_losses[key], 
                model.trainable_variables)
            self.optimizers[key+'_opt'].apply_gradients(
                zip(gradients, model.trainable_variables))
    
    def eval_step(self, x_real):
        losses, images = self.forward(x_real)
        template = '{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}'
        logging_message = template.format(*losses.values())
        return logging_message, images
    
    @tf.function
    def get_latent_code(self, x):
        z_real_mu, _ = self.enc(x, training=True)
        return z_real_mu

    @tf.function
    def generate(self, N):
        z = tf.random.normal([N, self.latent_dim])
        xi = tf.random.normal([N, self.latent_dim])
        x = self.gen(tf.concat([z, xi], axis=1), training=True)
        return x

    @tf.function
    def decode(self, z, xi):
        x = self.gen(tf.concat([z, xi], axis=1), training=True)
        return x
