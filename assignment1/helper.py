# helper functions
import numpy as np

def one_hot_encode(y: np.array, n_class: int):      # (B,1)
    b_size = y.shape[0]
    encoded = np.zeros((b_size, n_class))
    encoded[np.arange(b_size), y] = 1
    return encoded

class DataLoader:
    def __init__(self, xtrain, ytrain, batch_size):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.batch_size = batch_size
    def __call__(self):
        ix = np.random.randint(0, self.xtrain.shape[0], (self.batch_size,))
        xb = self.xtrain[ix]
        yb = self.ytrain[ix]
        return xb, yb


