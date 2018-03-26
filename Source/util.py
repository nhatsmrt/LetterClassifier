import numpy as np

def accuracy(predictions, y_true_enc):
    correct_prediction = np.equal(np.argmax(predictions, axis=1), np.argmax(y_true_enc, axis=1))
    accuracy = np.mean(np.cast(correct_prediction, np.float32), axis = -1)
    return accuracy