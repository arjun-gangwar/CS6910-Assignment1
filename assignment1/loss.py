import numpy as np

class CrossEntropyLoss:
    def __init__(self, epsilon):
        self.epsilon = epsilon
    def __call__(self, y_enc, y_hat):       # (B,), (B, classes)
        self.y_enc = y_enc
        self.y = np.argmax(y_enc, axis=-1)
        self.y_hat = y_hat
        self.out = np.mean(-np.log(y_hat[np.arange(y_enc.shape[0]), self.y] + self.epsilon))
        return self.out
    def diff(self):
        return (-1/(self.y_hat + self.epsilon)) * self.y_enc 
    def parameters(self):
        return []
    def d_parameters(self):
        return []
    
class MeanSquareLoss:
    def __init__(self, epsilon):
        self.epsilon = epsilon
    def __call__(self, y_enc, y_hat):       # (B,), (B, classes)
        self.y_enc = y_enc
        self.y = np.argmax(y_enc, axis=-1)
        self.y_hat = y_hat
        self.out = np.mean(np.square(self.y_enc - self.y_hat)) / 2
        return self.out
    def diff(self):
        return -1 * (self.y_enc - self.y_hat)
    def parameters(self):
        return []
    def d_parameters(self):
        return []