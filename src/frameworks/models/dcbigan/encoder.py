from __future__ import absolute_import, division, print_function

import tensorflow as tf

from math import sqrt
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv2D, Flatten, Dense, InputLayer)
from .blocks import DownBlock

class LogSigma(tf.keras.layers.Layer):
    def __init__(self, latent_dim, *args, **kwargs):
        self.latent_dim = latent_dim 
        super(LogSigma, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(LogSigma, self).build(input_shape)

        self.log_sigma = self.add_weight('log_sigma',
            shape=self.latent_dim,
            initializer=tf.initializers.constant(0),
            trainable=True)

    def call(self, inputs, training=True, noise=False):
        return self.log_sigma

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.latent_dim]

class Encoder(Model):
    def __init__(self, latent_dim, img_size, ch=64, **kwargs):
        name = kwargs.pop('name', 'encoder') 
        self.img_size = img_size
        super(Encoder, self).__init__(**kwargs, name=name)
        small = (img_size[0] == 32)  
        
        self.input_layer = InputLayer(img_size,
            name='{}_input'.format(name))
        self.from_rgb = Conv2D(ch, (5,5), (2, 2), 'SAME',
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_from_rgb'.format(name))
        
        self.block_1 = DownBlock(ch*2,
            name='{}_block_1'.format(name))
        self.block_2 = DownBlock(ch*4, (1,1) if small else (2,2),
            name='{}_block_2'.format(name))
        self.block_3 = DownBlock(ch*8,
            name='{}_block_3'.format(name))

        self.reshape = Flatten(
            name='{}_reshape'.format(name))
        self.dense_mu = Dense(latent_dim,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02), #stddev=0.1/sqrt(ch*8*16)), 
            name='{}_dense_mu'.format(name))
        self.log_sigma = LogSigma(latent_dim,
            name='{}_log_sigma'.format(name))
    
    def call(self, inputs, training=False):
        net = self.input_layer(inputs)
        net = self.from_rgb(net)
        net = self.block_1(net, training=training)
        net = self.block_2(net, training=training)
        net = self.block_3(net, training=training)
        net = self.reshape(net)
        mu = self.dense_mu(net)
        log_sigma = self.log_sigma(net)
        return mu, log_sigma