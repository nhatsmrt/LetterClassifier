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

# Built by Nhat Pham


class SimpleConvnet:
    def __init__(self, inp_w, inp_h, inp_d, n_classes = 26, keep_prob = 0.5, use_gpu = False):
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
        self._is_training = tf.placeholder(tf.bool)
        self._X = tf.placeholder(shape=[None, inp_w, inp_h, inp_d], dtype=tf.float32)
        self._X_norm = tf.contrib.layers.batch_norm(self._X, is_training=self._is_training)

        # Convolutional layers:
        # self._conv_module1 = self.convolutional_module_with_max_pool(self._X_norm, inp_channel=inp_d,
        #                                                              op_channel=64, name="module1")
        # self._conv_module2 = self.convolutional_module_with_max_pool(self._conv_module1, inp_channel=64,
        #                                                              op_channel=128, name="module2")

        self._conv1 = self.convolutional_layer(self._X_norm, "conv1", inp_channel = 1, op_channel = 64)
        self._conv2 = self.convolutional_layer(self._conv1, "conv2", inp_channel = 64, op_channel = 128, dropout = True)
        self._conv2_max_pool = self.max_pool_2x2(self._conv2)  # shape: [batch_size, 14, 14, 128]


        # Flatten:
        self._conv2_flat = tf.reshape(self._conv2_max_pool, shape=[-1, 6272], name = "flat")

        # Feedforward layers:
        self._fc1 = self.feed_forward(self._conv2_flat, inp_channel = 6272, op_channel = 26, op_layer=True,
                                      name="fc1")
        self._op_prob = tf.nn.softmax(self._fc1, name="prob")

    def ret_op(self):
        return self._op_prob

    def run_model(self, session, predict, loss_val, Xd, yd,
                  epochs=1, batch_size=1, print_every=1,
                  training=None, plot_losses=False, weight_save_path = None):
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

        # counter
        iter_cnt = 0
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
                feed_dict = {self._X: Xd[idx, :],
                             self._y: yd[idx],
                             self._is_training: training_now}
                # get batch size
                actual_batch_size = yd[idx].shape[0]

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
                if training_now and weight_save_path is not None:
                    save_path = saver.save(session, save_path = weight_save_path)
                    print("Model's weights saved at %s" % save_path)
                iter_cnt += 1
            total_correct = correct / Xd.shape[0]
            total_loss = np.sum(losses) / Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
                  .format(total_loss, total_correct, e + 1))
            # if plot_losses:
            #     plt.plot(losses)
            #     plt.grid(True)
            #     plt.title('Epoch {} Loss'.format(e + 1))
            #     plt.xlabel('minibatch number')
            #     plt.ylabel('minibatch loss')
            #     plt.show()
        return total_loss, total_correct


    # Define a max pool layer with size 2x2, stride of 2 and same padding.

    # Predict:
    def predict(self, X):
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                tf.global_variables_initializer().run()
                ans = sess.run(self._op_prob, feed_dict = {self._X : X, self._is_training : False})
                return ans

    # Define layers and modules:
    def convolutional_layer(self, x, name, inp_channel, op_channel, kernel_size = 3, strides = 1, padding = 'VALID', pad = 1, dropout = False, not_activated = False):
        if pad != 0:
            x_padded = tf.pad(x, self.create_pad(4, pad))
        else:
            x_padded = x
        W_conv = tf.get_variable("W" + name, shape = [kernel_size, kernel_size, inp_channel, op_channel], initializer = tf.keras.initializers.he_normal())
        b_conv = tf.get_variable("b" + name, initializer = tf.zeros(op_channel))
        z_conv = tf.nn.conv2d(x_padded, W_conv, strides = [1, strides, strides, 1], padding = padding) + b_conv
        a_conv = tf.nn.relu(z_conv)
        # h_conv = tf.layers.batch_normalization(a_conv, axis = 1, training = self._is_training)
        if dropout:
            a_conv_dropout = tf.nn.dropout(a_conv, keep_prob = self._keep_prob)
            return a_conv_dropout
        if not_activated:
            return z_conv
        return a_conv

    def convolutional_module(self, x, name, inp_channel, op_channel, down_rate = 2):
        conv1 = self.convolutional_layer(x, name + "conv1", inp_channel, op_channel)
        conv2 = self.convolutional_layer(conv1, name + "conv2", op_channel, op_channel, strides = down_rate)
        # conv3 = self.convolutional_layer(conv2, name + "conv3", inp_channel, op_channel, dropout = True)

        batch_norm = tf.contrib.layers.batch_norm(conv2, is_training = self._is_training)

        return batch_norm

    def convolutional_module_with_max_pool(self, x, inp_channel, op_channel, name):
        # conv1 = self.convolutional_layer(x, inp_channel = inp_channel, op_channel = op_channel, name = name + "_conv1")
        conv1 = self.convolutional_layer(x, inp_channel = inp_channel, op_channel = op_channel, name = name + "_conv1", dropout = True)
        conv2 = self.convolutional_layer(conv1, inp_channel = op_channel, op_channel = op_channel, name = name + "_conv2", dropout = True)
        conv2_max_pool = self.max_pool_2x2(conv2)

        return conv2_max_pool


    def residual_module(self, x, name, inp_channel):
        conv1 = self.convolutional_layer(x, name + "conv1", inp_channel, inp_channel)
        conv2 = self.convolutional_layer(conv1, name + "conv2", inp_channel, inp_channel, not_activated = True)
        # conv3 = self.convolutional_layer(conv2, name + "conv3", inp_channel, op_channel, dropout = True)
        res_layer = tf.nn.relu(tf.add(conv2, x, name = "res"))

        batch_norm = tf.contrib.layers.batch_norm(res_layer, is_training = self._is_training)

        return batch_norm

    def feed_forward(self, x, name, inp_channel, op_channel, op_layer = False):
        W = tf.get_variable("W_" + name, shape = [inp_channel, op_channel], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b_" + name, shape = [op_channel],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        z = tf.matmul(x, W) + b
        if op_layer:
            # a = tf.nn.sigmoid(z)
            # return a
            return z
        else:
            a = tf.nn.relu(z)
            a_norm = tf.layers.batch_normalization(a, training = self._is_training)
            return a_norm

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def global_average_pooling(self, x):
        return tf.reduce_mean(x, axis = [1, 2])


    # Train:
    def fit(self, X, y, num_epoch = 64, batch_size = 16, weight_save_path = None, weight_load_path = None):
        self._y = tf.placeholder(tf.float32, shape = [None, self._n_classes])
        self._mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self._fc1, labels = self._y))
        self._optimizer = tf.train.AdamOptimizer(1e-4)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self._train_step = self._optimizer.minimize(self._mean_loss)
        self._sess = tf.Session()
        if weight_load_path is not None:
            loader = tf.train.Saver()
            loader.restore(sess = self._sess, save_path = weight_load_path)
            print("Weight loaded successfully")
        else:
            self._sess.run(tf.global_variables_initializer())
        if num_epoch > 0:
            print('Training Characters Classifier for ' + str(num_epoch) +  ' epochs')
            self.run_model(self._sess, self._op_prob, self._mean_loss, X, y, num_epoch, batch_size, 1, self._train_step, weight_save_path = weight_save_path)



    def create_pad(self, n, pad):
        pad_matrix = [[0, 0]]
        for i in range(n-2):
            pad_matrix.append([pad, pad])
        pad_matrix.append([0, 0])
        return tf.constant(pad_matrix)




    def evaluate (self, X, y):
        self.run_model(self._sess, self._op_prob, self._mean_loss, X, y, 1, 16)
