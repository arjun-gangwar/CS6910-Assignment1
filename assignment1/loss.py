import numpy as np

class CrossEntropyLoss:
    def __call__(self, y_enc, y_hat):       # (B,), (B, classes)
        self.y_enc = y_enc
        self.y = np.argmax(y_enc, axis=-1)
        self.y_hat = y_hat
        self.out = np.mean(-np.log(y_hat[np.arange(y_enc.shape[0]), self.y]))
        return self.out
    def diff(self):
        return (-1/(self.y_hat)) * self.y_enc 
    def parameters(self):
        return []
    def d_parameters(self):
        return []