import numpy as np

class Linear:
    def __init__(self, fan_in, fan_out, init_method):
        if init_method == "xavier":
            self.weight = np.random.randn(fan_in, fan_out)
            self.bias = np.zeros(fan_out)
        else:
            self.weight = np.random.randn(fan_in, fan_out)
            self.bias = np.zeros(fan_out)
        self.dw = 0.
        self.db = 0.
        self.uw = 0.
        self.ub = 0.
        self.nesw = self.weight
        self.nesb = self.bias
    def __call__(self, x):      # (B x fan_in) x (fan_in x fan_out)
        self.x = x
        self.out = x @ self.weight + self.bias
        return self.out
    def diff(self, prev_grad):
        diff = self.weight.T 

        # calculate dl/dw, dl/db ---> can write a separate func in class
        self.dw = np.mean(self.x[:,:,np.newaxis] @ prev_grad[:,np.newaxis,:], axis=0)
        self.db = np.mean(prev_grad, axis=0)
        
        return prev_grad @ diff
    
    def parameters(self):
        return [self.weight, self.bias]
    def d_parameters(self):
        return [self.dw, self.db]