import tensorflow as tf
import numpy as np
import pandas as pd
import path, os, random, math
from pathlib import Path

from Source import SimpleConvnet, accuracy

from sklearn.preprocessing import LabelBinarizer


# Define paths:
d = Path().resolve()
# source_path = str(d) + "/Source/"
data_path = str(d) + "/Data"
train_path = data_path + "/train.csv"
predictions_path = "/output/"
sample_path = data_path + "/sample_submission.csv"

# Prepare data:
df_train = pd.read_csv(data_path + "/emnist-letters-train.csv")
df_test = pd.read_csv(data_path + "/emnist-letters-test.csv")

y_train = df_train.iloc[:, 0].values
y_test = df_test.iloc[:, 0].values


X_train = df_train.iloc[:, 1:].values
X_test = df_test.iloc[:, 1:].values
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode the y-values:
lb = LabelBinarizer()
lb.fit(y_train)
y_train_enc = lb.transform(y_train)
y_test_enc = lb.transform(y_test)


# Define hyperparameters:
num_epoch = 1
batch_size = 16




# Define model:
model = SimpleConvnet(inp_w = 28, inp_h = 28, inp_d = 1)
model.fit(X_train, y_train_enc, num_epoch = num_epoch, batch_size = batch_size)

# Test model:
predictions = model.predict(X_test)
correct_prediction = accuracy(predictions, y_test_enc)
accuracy = np.mean(np.cast(correct_prediction, np.float32) ,axis = -1)
