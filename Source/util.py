import numpy as np

def accuracy(predictions, y_true_enc):
    correct_prediction = np.equal(np.argmax(predictions, axis = 1), np.argmax(y_true_enc, axis = 1))
    print(correct_prediction)
    accuracy = np.mean(correct_prediction.astype(np.float32))
    return accuracy