import numpy as np

class Sigmoid:
    def __call__(self, x):
        self.x = x
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    def diff(self, prev_grad):
        diff = (self.out * (1 - self.out))
        return prev_grad * diff
    def parameters(self):
        return []
    def d_parameters(self):
        return []
    
class Tanh:
    def __call__(self, x):
        self.x = x
        self.out = np.tanh(x)
        return self.out
    def diff(self, prev_grad):
        diff = 1 - np.square(self.out)
        return prev_grad * diff
    def parameters(self):
        return []
    def d_parameters(self):
        return []
    
class ReLU:
    def __call__(self, x):
        self.x = x
        self.out = np.maximum(0, x)
        return self.out
    def diff(self, prev_grad):
        diff = (self.out > 0).astype(int)
        return prev_grad * diff
    def parameters(self):
        return []
    def d_parameters(self):
        return []

class Softmax:
    def __call__(self, x):
        self.x = x
        # self.out = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shifted)
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out
    def diff(self, prev_grad):
        e_l = prev_grad != 0.
        diff = np.sum(self.out * e_l, axis=-1, keepdims=True) * (e_l - self.out)
        return np.sum(prev_grad, axis=-1, keepdims=True) * diff
    def parameters(self):
        return []
    def d_parameters(self):
        return []
    
class SoftmaxMSE:
    def __call__(self, x):
        self.x = x
        # self.out = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shifted)
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out
    def diff(self, prev_grad, y_enc):
        diff = np.sum(self.out * y_enc, axis=-1, keepdims=True) * (y_enc - self.out)
        return prev_grad * diff
    def parameters(self):
        return []
    def d_parameters(self):
        return []