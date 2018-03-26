import numpy as np

def accuracy(predictions, y_true_enc):
    return np.equal(np.argmax(predictions, axis=1), np.argmax(y_true_enc, axis=1))