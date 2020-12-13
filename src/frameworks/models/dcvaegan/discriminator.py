from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv2D, Flatten, Dense, InputLayer)
from .blocks import DownBlock
from numpy import sqrt


class Discriminator(Model):
    def __init__(self, img_size, ch=64, **kwargs):
        name = kwargs.pop('name', 'discriminator')  
        super(Discriminator, self).__init__(**kwargs, name=name)
        self.small = (img_size[0] == 32)  
        self.ch = ch

        self.input_layer = InputLayer(img_size,
            name='{}_input'.format(name))
        self.from_rgb = Conv2D(ch, (5,5), (2, 2), 'SAME',
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_from_rgb'.format(name))
        
        self.block_1 = DownBlock(ch*2,
            name='{}_block_1'.format(name))
        self.block_2 = DownBlock(ch*4, (1,1) if self.small else (2,2),
            name='{}_block_2'.format(name))
        self.block_3 = DownBlock(ch*8,
            name='{}_block_3'.format(name))

        self.reshape = Flatten(
            name='{}_reshape'.format(name))
        self.project = Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_project'.format(name))
    
    def call(self, inputs, training=False):
        net = self.input_layer(inputs)
        net = self.from_rgb(net)
        net = self.block_1(net, training=training)
        net = self.block_2(net, training=training)
        if not self.small:
            z = net / sqrt(2)
        net = self.block_3(net, training=training)
        if self.small:
            z = net
        net = self.reshape(net)
        net = self.project(net)
        return net, z