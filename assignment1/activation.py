import numpy as np

class Sigmoid:
    def __call__(self, x):
        self.x = x      # for backprop
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    def diff(self, prev_grad):
        diff = (self.out * (1 - self.out))
        return prev_grad * diff
    def parameters(self):
        return []
    def d_parameters(self):
        return []

class Softmax:
    def __call__(self, x):
        self.x = x      # for backprop
        self.out = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return self.out
    def diff(self, prev_grad):
        e_l = prev_grad != 0.
        diff = np.sum(self.out * e_l, axis=-1, keepdims=True) * (e_l - self.out)
        return prev_grad * diff
    def parameters(self):
        return []
    def d_parameters(self):
        return []