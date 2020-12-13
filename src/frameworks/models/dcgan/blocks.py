from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv2D, Conv2DTranspose)



class UpBlock(Model):
    def __init__(self, ch, stride=2, **kwargs):
        name = kwargs.pop('name', 'up_block')
        activation = kwargs.pop('activation', tf.nn.relu)
        super(UpBlock, self).__init__(**kwargs, name=name)

        self.conv_transpose = Conv2DTranspose(ch, (5,5), stride, 'SAME', 
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_conv_resize'.format(name))
        self.batch_norm = BatchNormalization(
            momentum=0.9,
            gamma_initializer=tf.random_normal_initializer(1.0, stddev=0.02),
            name='{}_batch_normalization'.format(name))
        self.activation = Activation(activation,
            name='{}_activation'.format(name))

    def call(self, inputs, training=False):
        net = inputs
        net = self.conv_transpose(net)
        net = self.batch_norm(net, training=training)
        net = self.activation(net)
        return net


class DownBlock(Model):
    def __init__(self, ch, stride=2, **kwargs):
        name = kwargs.pop('name', 'down_block')
        activation = kwargs.pop('activation', tf.nn.leaky_relu)
        super(DownBlock, self).__init__(**kwargs, name=name)

        self.conv = Conv2D(ch, (5,5), stride, 'SAME', 
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            name='{}_conv_resize'.format(name))
        self.batch_norm = BatchNormalization(
            momentum=0.9,
            gamma_initializer=tf.random_normal_initializer(1.0, stddev=0.02),
            name='{}_batch_normalization'.format(name))
        self.activation = Activation(activation,
            name='{}_activation'.format(name))

    def call(self, inputs, training=False):
        net = inputs
        net = self.conv(net)
        net = self.batch_norm(net, training=training)
        net = self.activation(net)
        return net