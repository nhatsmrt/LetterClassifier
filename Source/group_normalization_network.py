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

class FourthModel(SimpleConvnet):
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
        # self._X_norm = tf.layers.batch_normalization(self._X, training = self._is_training)
        self._X_norm = self.group_normalization(self._X, name = "X_norm", G = 1, inp_channel = inp_d)

        # Convolutional and max-pool:
        self._convolution_layer1 = self.convolutional_layer(self._X_norm, kernel_size = 7, inp_channel = inp_d, op_channel = 64, name = "conv_layer1", strides = 2, padding = 'SAME')
        self._convolution_layer1_max_pool = tf.nn.max_pool(self._convolution_layer1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')


        # Residual Modules:
        self._res_module1 = self.residual_module(self._convolution_layer1_max_pool, inp_channel = 64, name = "res_module1")
        self._res_module2 = self.residual_module(self._res_module1, inp_channel = 64, name = "res_module2")

        self._convolution_layer2 = self.convolutional_layer(self._res_module2, kernel_size = 7, inp_channel = 64, op_channel = 128, name = "conv_layer2", strides = 2, padding = 'SAME')
        self._convolution_layer2_max_pool = tf.nn.max_pool(self._convolution_layer2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        self._res_module3 = self.residual_module(self._convolution_layer2_max_pool, inp_channel = 128, name = "res_module3")
        self._res_module4 = self.residual_module(self._res_module3, inp_channel = 128, name = "res_module4")



        # Flatten:
        # self._conv_module2_dropout = tf.nn.dropout(self._conv_module2, keep_prob = self._keep_prob)
        self._flat = tf.reshape(self._res_module4, [-1, 1152], name = "flat")
        # self._op = self.feed_forward(self._flat, name = "op", inp_channel = 6272, op_channel = 26)
        self._fc1 = self.feed_forward(self._flat, name = "fc1", inp_channel = 1152, op_channel = 100)
        self._fc2 = self.feed_forward(self._fc1, inp_channel = 100, op_channel = 26, name = "fc2", op_layer = True)
        self._op = tf.nn.dropout(self._fc2, keep_prob = self._keep_prob_tensor)

        self._op_prob = tf.nn.softmax(self._op, name = "prob")

    def ret_op(self):
        return self._op_prob

# Adapt from Stanford's CS231n Assignment3
    def run_model(self, session, predict, loss_val, Xd, yd,
                  epochs=1, batch_size=1, print_every=1,
                  training=None, plot_losses=False, weight_save_path = None, patience = None):
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.argmax(self._op_prob, axis = 1), tf.argmax(self._y, axis = 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Define saver:
        saver = tf.train.Saver()

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self._mean_loss, correct_prediction, accuracy]
        if training_now:
            variables[-1] = training
            self._keep_prob_passed = self._keep_prob
        else:
            self._keep_prob_passed = 1.0

        # counter
        iter_cnt = 0
        val_losses = []
        early_stopping_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size) % Xd.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                # create a feed dictionary for this batch
                # get batch size
                actual_batch_size = yd[idx].shape[0]

                if i < int(math.ceil(Xd.shape[0] / batch_size)) - 1:
                    feed_dict = {self._X: Xd[idx, :],
                                 self._y: yd[idx],
                                 self._is_training: training_now,
                                 self._keep_prob_tensor: self._keep_prob_passed}
                    # have tensorflow compute loss and correct predictions
                    # and (if given) perform a training step
                    loss, corr, _ = session.run(variables, feed_dict=feed_dict)

                    # aggregate performance stats
                    losses.append(loss * actual_batch_size)
                    correct += np.sum(corr)

                    # print every now and then
                    if training_now and (iter_cnt % print_every) == 0:
                        print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                              .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
                else:
                    feed_dict = {self._X: Xd[idx, :],
                                 self._y: yd[idx],
                                 self._is_training: False,
                                 self._keep_prob_tensor: 1.0}
                    val_loss = session.run(self._mean_loss, feed_dict = feed_dict)
                    print("Validation loss: " + str(val_loss))
                    val_losses.append(val_loss)
                    # if training_now and weight_save_path is not None:
                    if training_now and val_loss <= min(val_losses) and weight_save_path is not None:
                        save_path = saver.save(session, save_path = weight_save_path)
                        print("Model's weights saved at %s" % save_path)
                    if patience is not None:
                        if val_loss > min(val_losses):
                            early_stopping_cnt += 1
                        else:
                            early_stopping_cnt = 0
                        if early_stopping_cnt > patience:
                            print("Patience exceeded. Finish training")
                            return
                iter_cnt += 1
            total_correct = correct / Xd.shape[0]
            total_loss = np.sum(losses) / Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
                  .format(total_loss, total_correct, e + 1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e + 1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss, total_correct

    # Define layers and modules:
    def convolutional_layer(self, x, name, inp_channel, op_channel, kernel_size = 3, strides = 1, padding = 'VALID', pad = 1, dropout = False, not_activated = False):
        if pad != 0:
            x_padded = tf.pad(x, self.create_pad(4, pad))
        else:
            x_padded = x
        W_conv = tf.get_variable("W_" + name, shape = [kernel_size, kernel_size, inp_channel, op_channel], initializer = tf.keras.initializers.he_normal())
        b_conv = tf.get_variable("b_" + name, initializer = tf.zeros(op_channel))
        z_conv = tf.nn.conv2d(x_padded, W_conv, strides = [1, strides, strides, 1], padding = padding) + b_conv
        a_conv = tf.nn.relu(z_conv)
        h_conv = self.group_normalization(a_conv, name = name + "_norm", inp_channel = op_channel, G = 32)
        if dropout:
            a_conv_dropout = tf.nn.dropout(a_conv, keep_prob = self._keep_prob)
            return a_conv_dropout
        if not_activated:
            return z_conv
        return h_conv

    def residual_module(self, x, name, inp_channel):
        conv1 = self.convolutional_layer(x, name + "_conv1", inp_channel, inp_channel, not_activated = True)
        batch_norm_1 = self.group_normalization(conv1, inp_channel= inp_channel, name = name + "_batch_norm_1", G = 32)
        z_1 = tf.nn.relu(batch_norm_1)
        conv2 = self.convolutional_layer(z_1, name + "_conv2", inp_channel, inp_channel, not_activated = True)
        batch_norm_2 = self.group_normalization(conv2, inp_channel = inp_channel, name = name + "_batch_norm_2", G = 32)
        res_layer = tf.nn.relu(tf.add(batch_norm_2, x, name = name + "res"))


        return res_layer

    def feed_forward(self, x, name, inp_channel, op_channel, op_layer = False):
        W = tf.get_variable("W_" + name, shape = [inp_channel, op_channel], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b_" + name, shape = [op_channel],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        z = tf.matmul(x, W) + b
        if op_layer:
            # a = tf.nn.sigmoid(z)
            # return a
            return tf.layers.batch_normalization(z, training = self._is_training)
        else:
            a = tf.nn.relu(z)
            a_norm = tf.layers.batch_normalization(a, training = self._is_training)
            return a_norm

    def group_normalization(self, x, name, inp_channel, G, eps = 1e-5):
        shape_x = tf.shape(x) # N, H, W, C
        gamma = tf.get_variable(name = name + "_gamma", shape = [1, 1, 1, inp_channel])
        beta = tf.get_variable(name = name + "_beta", shape = [1, 1, 1, inp_channel])
        x_grouped = tf.reshape(x, [shape_x[0], shape_x[1], shape_x[2], G, shape_x[3] // G])

        mean, var = tf.nn.moments(x_grouped, axes = [1, 2, 4], keep_dims = True)
        x_norm = (x_grouped - mean) / tf.sqrt(var + eps)

        x_norm_reshaped = tf.reshape(x_norm, [shape_x[0], shape_x[1], shape_x[2], shape_x[3]])

        return x_norm_reshaped * gamma + beta




