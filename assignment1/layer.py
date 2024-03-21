import numpy as np

class Linear:
    def __init__(self, fan_in, fan_out, init_method):
        if init_method == "xavier":
            xav = np.sqrt(6.0/(fan_in + fan_out))
            self.weight = np.random.normal(loc=0, scale=xav, size=(fan_in, fan_out))
            self.bias = np.zeros(fan_out)
        else:
            self.weight = np.random.randn(fan_in, fan_out)
            self.bias = np.zeros(fan_out)
        self.dw = None
        self.db = None
        self.uw = 0.
        self.ub = 0.
    def __call__(self, x):      # (B x fan_in) x (fan_in x fan_out)
        self.x = x
        self.out = x @ self.weight + self.bias
        return self.out
    def diff(self, prev_grad):
        diff = self.weight.T 
        # calculate dl/dw, dl/db ---> can write a separate func in class
        self.dw = np.dot(self.x.T, prev_grad)
        self.db = np.sum(prev_grad, axis=0)
        return prev_grad @ diff
    
    def parameters(self):
        return [self.weight, self.bias]
    def d_parameters(self):
        return [self.dw, self.db]