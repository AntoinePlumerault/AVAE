from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv2DTranspose, Reshape, Dense, InputLayer)
from .blocks import UpBlock



class Generator(Model):
    def __init__(self, latent_dim, img_size, ch=64, **kwargs):
        name = kwargs.pop('name', 'generator') 
        self.img_size = img_size
        super(Generator, self).__init__(**kwargs, name=name)
        small = (img_size[0] == 32)  

        self.input_layer = InputLayer([latent_dim],
            name='{}_input'.format(name))
        self.project = Dense(ch*8*4*4, use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_project'.format(name))
        self.reshape = Reshape([4,4,ch*8],
            name='{}_reshape'.format(name))
        self.batch_norm = BatchNormalization(
            momentum=0.9,
            gamma_initializer=tf.random_normal_initializer(1.0, stddev=0.02),
            name='{}_batch_norm'.format(name))
        self.activation = Activation(tf.nn.relu,
            name='{}_activation'.format(name))
        
        self.block_1 = UpBlock(ch*4,
            name='{}_block_1'.format(name))
        self.block_2 = UpBlock(ch*2, (1,1) if small else (2,2),
            name='{}_block_2'.format(name))
        self.block_3 = UpBlock(ch,
            name='{}_block_3'.format(name))
        
        self.to_rgb = Conv2DTranspose(img_size[-1], (5,5), (2, 2), 'SAME', 
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            activation=tf.math.tanh,
            name='{}_to_rgb'.format(name))
    
    def call(self, inputs, training=False):
        net = self.input_layer(inputs)
        net = self.project(net)
        net = self.reshape(net)
        net = self.batch_norm(net, training=training)
        net = self.activation(net)
        net = self.block_1(net, training=training)
        net = self.block_2(net, training=training)
        net = self.block_3(net, training=training)
        net = self.to_rgb(net)
        return net