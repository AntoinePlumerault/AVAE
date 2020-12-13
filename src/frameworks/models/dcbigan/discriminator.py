from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv2D, Flatten, Dense, InputLayer, Concatenate)
from .blocks import DownBlock


class X_net(Model):
    def __init__(self, latent_dim, img_size, ch=64, **kwargs):
        name = kwargs.pop('name', 'x_net')  
        super(X_net, self).__init__(**kwargs, name=name)
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
        
        self.output_layer = Dense(ch*8,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_from_rgb'.format(name))

    def call(self, inputs, training=False):
        net = self.input_layer(inputs)
        net = self.from_rgb(net)
        net = self.block_1(net, training=training)
        net = self.block_2(net, training=training)
        net = self.block_3(net, training=training)
        net = self.reshape(net)
        net = self.output_layer(net)
        
        return net

class Z_net(Model):
    def __init__(self, latent_dim, ch=64, **kwargs):
        name = kwargs.pop('name', 'z_net')  
        super(Z_net, self).__init__(**kwargs, name=name) 

        self.input_layer = InputLayer(latent_dim,
            name='{}_input'.format(name))
        self.hidden_layer_1 = Dense(ch*8,
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_hidden_layer_1'.format(name))
        self.hidden_layer_2 = Dense(ch*8,
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_hidden_layer_2'.format(name))
        self.output_layer = Dense(ch*8,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_from_rgb'.format(name))
        
    def call(self, inputs, training=False):
        net = self.input_layer(inputs)
        net = self.hidden_layer_1(net)
        net = self.hidden_layer_2(net)
        net = self.output_layer(net)
        
        return net

class Discriminator(Model):
    def __init__(self, latent_dim, img_size, ch=64, **kwargs):
        name = kwargs.pop('name', 'discriminator')  
        super(Discriminator, self).__init__(**kwargs, name=name)
        small = (img_size[0] == 32)  

        self.x_input_layer = InputLayer(img_size,
            name='{}_x_input'.format(name))
        self.z_input_layer = InputLayer(latent_dim,
            name='{}_z_input'.format(name))
        
        self.x_net = X_net(latent_dim, img_size, ch,
            name='{}_x_net'.format(name))
        self.z_net = Z_net(latent_dim, ch,
            name='{}_z_net'.format(name))
        
        self.x_dense = Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_x_dense'.format(name))
        self.z_dense = Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_z_dense'.format(name))
        
        self.xz_concat = Concatenate(
            name='{}_concat'.format(name))
        self.xz_hidden_1 = Dense(ch*16,
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_xz_hidden_1'.format(name))
        self.xz_hidden_2 = Dense(ch*16,
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_xz_hidden_2'.format(name))
        self.xz_output = Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_xz_output'.format(name))

    def call(self, inputs, training=False):
        x, z = inputs
        x_net = self.x_input_layer(x)
        x_net = self.x_net(x_net,training=training)
        
        z_net = self.z_input_layer(z)
        z_net = self.z_net(z_net,training=training)
        
        xz_net = self.xz_concat([x_net, z_net])
        xz_net = self.xz_hidden_1(xz_net)
        xz_net = self.xz_hidden_2(xz_net)
        xz_net = self.xz_output(xz_net)
        
        x_net = self.x_dense(x_net)
        z_net = self.z_dense(z_net)

        return x_net, z_net, xz_net 