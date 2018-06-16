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
from .deep_model import DeeperNet

import matplotlib.pyplot as plt

class SimpleAveragingEnsembler:

    def __init__(self, n_models, weights_list, inp_w, inp_h, inp_d, n_classes = 26):
        self._inp_w = inp_w
        self._inp_h = inp_h
        self._inp_d = inp_d
        self._n_classes = n_classes
        self._n_models = n_models
        self._weights_list = weights_list

    def predict(self, X_test):
        predictions_list = []
        X_train_dummy = np.zeros(shape = [1, self._inp_w, self._inp_h, self._inp_d])
        X_train_dummy = np.zeros(shape = [1, self._n_classes])
        for i in range(self._n_models):
            tf.reset_default_graph()
            model = DeeperNet(self._inp_w, self._inp_h, self._inp_d, self._n_classes)
            model.fit(X_train_dummy, X_train_dummy, num_epoch= 0, batch_size = 16,
                       weight_load_path = self._weights_list[i])
            predictions_list.append(model.predict(X_test))
        predictions_list = np.array(predictions_list)
        return np.mean(predictions_list, axis = 0)




