import tensorflow as tf
import numpy as np
import scipy.misc
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
import timeit
import math
import os
import json
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
from .simple_covnet import SimpleConvnet

import matplotlib.pyplot as plt

# Built by Nhat Hoang Pham

class SENet(SimpleConvnet):
    def __init__(self, inp_w, inp_h, inp_d, n_classes = 26, keep_prob = 0.8, use_gpu = False):
        self._n_classes = n_classes
        self._keep_prob = keep_prob
        self._use_gpu = use_gpu
        if use_gpu:
            with tf.device('/device:GPU:0'):
                self.create_network(inp_w, inp_h, inp_d)
        else:
            with tf.device('/device:CPU:0'):
                self.create_network(inp_w, inp_h, inp_d)


    def create_network(self, inp_w, inp_h, inp_d):
        self._keep_prob_tensor = tf.placeholder(tf.float32)
        self._is_training = tf.placeholder(tf.bool)
        self._X = tf.placeholder(shape=[None, inp_w, inp_h, inp_d], dtype=tf.float32)
        # self._X_norm = tf.contrib.layers.batch_norm(self._X, is_training=self._is_training)
        self._X_norm = tf.layers.batch_normalization(self._X, training = self._is_training)

        # Convolutional and max-pool:
        self._convolution_layer1 = self.convolutional_layer(self._X_norm, kernel_size = 7, inp_channel = inp_d, op_channel = 64, name = "conv_layer1", strides = 2, padding = 'SAME')
        self._convolution_layer1_max_pool = tf.nn.max_pool(self._convolution_layer1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')


        # Residual Modules:
        self._res_module1 = self.residual_module_with_se(self._convolution_layer1_max_pool, inp_channel = 64, name = "res_module1")
        self._res_module2 = self.residual_module_with_se(self._res_module1, inp_channel = 64, name = "res_module2")

        self._convolution_layer2 = self.convolutional_layer(self._res_module2, kernel_size = 7, inp_channel = 64, op_channel = 128, name = "conv_layer2", strides = 2, padding = 'SAME')
        self._convolution_layer2_max_pool = tf.nn.max_pool(self._convolution_layer2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        self._res_module3 = self.residual_module_with_se(self._convolution_layer2_max_pool, inp_channel = 128, name = "res_module3")
        self._res_module4 = self.residual_module_with_se(self._res_module3, inp_channel = 128, name = "res_module4")



        # Flatten:
        # self._conv_module2_dropout = tf.nn.dropout(self._conv_module2, keep_prob = self._keep_prob)
        self._flat = tf.reshape(self._res_module4, [-1, 1152], name = "flat")
        # self._op = self.feed_forward(self._flat, name = "op", inp_channel = 6272, op_channel = 26)
        self._fc1 = self.feed_forward(self._flat, name = "fc1", inp_channel = 1152, op_channel = 100)
        self._fc2 = self.feed_forward(self._fc1, inp_channel = 100, op_channel = 26, name = "fc2", op_layer = True)
        self._op = tf.nn.dropout(self._fc2, keep_prob = self._keep_prob_tensor)

        self._op_prob = tf.nn.softmax(self._op, name = "prob")

    def residual_module_with_se(self, x, name, inp_channel):
        conv1 = self.convolutional_layer(x, name + "_conv1", inp_channel, inp_channel, not_activated = True)
        batch_norm_1 = tf.layers.batch_normalization(conv1, training = self._is_training)
        z_1 = tf.nn.relu(batch_norm_1)
        conv2 = self.convolutional_layer(z_1, name + "_conv2", inp_channel, inp_channel, not_activated = True)
        batch_norm_2 = tf.layers.batch_normalization(conv2, training = self._is_training)
        batch_norm_2_se = self.se_block(batch_norm_2, name = name + "_se", n_channels = inp_channel)
        res_layer = tf.nn.relu(tf.add(batch_norm_2_se, x, name = name + "res"))


        return res_layer



    def squeeze(self, x):
        return self.global_average_pooling(x)

    def excite(self, x, name, n_channels, reduction_ratio = 16):
        x_shape = tf.shape(x)
        W_1 = tf.get_variable(shape = [n_channels, n_channels // reduction_ratio], name = name + "_W1")
        z_1 = tf.nn.relu(tf.matmul(x, W_1))
        W_2 = tf.get_variable(shape = [n_channels // reduction_ratio, n_channels], name = name + "_W2")
        return tf.nn.sigmoid(tf.matmul(z_1, W_2))


    def se_block(self, x, name, n_channels):
        x_shape = tf.shape(x)
        x_squeezed = self.squeeze(x)
        x_excited = self.excite(x_squeezed, name = name + "_excited", n_channels = n_channels)
        x_excited_broadcasted = tf.reshape(x_excited, shape = [x_shape[0], 1, 1, x_shape[-1]])
        return tf.multiply(x, x_excited_broadcasted)
