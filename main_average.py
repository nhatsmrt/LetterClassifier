import tensorflow as tf
import numpy as np
import pandas as pd
import path, os, random, math
from pathlib import Path

from Source import accuracy, SimpleAveragingEnsembler

from sklearn.preprocessing import LabelBinarizer


# Define paths:
d = Path().resolve()
data_path = str(d) + "/Data"
weight_load_path1 = str(d) + "/weights/1/model_deeper.ckpt"
weight_load_path2 = str(d) + "/weights/2/model_deeper.ckpt"


# Prepare data:
df_test = pd.read_csv(data_path + "/emnist-letters-test.csv")

y_test = df_test.iloc[:, 0].values


X_test = df_test.iloc[:, 1:].values
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode the y-values:
lb = LabelBinarizer()
y_test_enc = lb.fit_transform(y_test)


# Define hyperparameters:
n_models = 2



# Define models:
weights_list = [weight_load_path1, weight_load_path2]
ensembler = SimpleAveragingEnsembler(n_models, inp_w = 28, inp_h = 28, inp_d = 1, weights_list = weights_list)

# Test model:
predictions = ensembler.predict(X_test)
print(accuracy(predictions, y_test_enc))
