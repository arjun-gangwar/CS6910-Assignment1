import numpy as np
from layer import Linear

class SGD():
    def __init__(self, layers, learning_rate, epsilon):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epsilon = epsilon
    # overriding abstract method
    def __call__(self, epoch):
        # gradient descent
        for k, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer.weight -= self.learning_rate * layer.dw 
                layer.bias -= self.learning_rate * layer.db

class Momentum():
    def __init__(self, layers, learning_rate, momentum, epsilon):
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epsilon = epsilon
    # overriding abstract method
    def __call__(self, epoch):
        # momentum based gradient descent
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.uw = self.momentum * layer.uw + layer.dw
                layer.ub = self.momentum * layer.ub + layer.db
                layer.weight -= self.learning_rate * layer.uw
                layer.bias -= self.learning_rate * layer.ub

class Nestrov():
    def __init__(self, layers, learning_rate, momentum, epsilon):
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epsilon = epsilon
    # overriding abstract method
    def __call__(self, epoch):
        # nesterov based gradient descent
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.uw = self.momentum * layer.uw + layer.dw
                layer.ub = self.momentum * layer.ub + layer.db

                layer.weight -= self.learning_rate * (self.momentum * layer.uw + layer.dw)
                layer.bias -= self.learning_rate * (self.momentum * layer.ub + layer.db)

class RMSProp():
    def __init__(self, layers, learning_rate, beta, epsilon):
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
    # overriding abstract method
    def __call__(self, epoch):
        # rmsprop based gradient descent
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.uw = self.beta * layer.uw + (1-self.beta) * np.square(layer.dw)
                layer.ub = self.beta * layer.ub + (1-self.beta) * np.square(layer.db)
                layer.weight -= (self.learning_rate * layer.dw) / (np.sqrt(layer.uw) + self.epsilon)
                layer.bias -= (self.learning_rate * layer.db) / (np.sqrt(layer.ub) + self.epsilon)

class Adam():
    def __init__(self, layers, learning_rate, beta1, beta2, epsilon):
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    # overriding abstract method
    def __call__(self, epoch):
        # adam based gradient descent
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.mw = self.beta1 * layer.mw + (1-self.beta1) * layer.dw
                layer.mb = self.beta1 * layer.mb + (1-self.beta1) * layer.db
                layer.uw = self.beta2 * layer.uw + (1-self.beta2) * np.square(layer.dw)
                layer.ub = self.beta2 * layer.ub + (1-self.beta2) * np.square(layer.db)

                # bias correcting
                mw_hat = layer.mw / (1-np.power(self.beta1, epoch))
                mb_hat = layer.mb / (1-np.power(self.beta1, epoch))
                uw_hat = layer.uw / (1-np.power(self.beta2, epoch))
                ub_hat = layer.ub / (1-np.power(self.beta2, epoch))

                layer.weight -= (self.learning_rate * mw_hat) / (np.sqrt(uw_hat) + self.epsilon)
                layer.bias -= (self.learning_rate * mb_hat) / (np.sqrt(ub_hat) + self.epsilon)

class NAdam():
    def __init__(self, layers, learning_rate, beta1, beta2, epsilon):
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    # overriding abstract method
    def __call__(self, epoch):
        # nadam based gradient descent
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.mw = self.beta1 * layer.mw + (1-self.beta1) * layer.dw
                layer.mb = self.beta1 * layer.mb + (1-self.beta1) * layer.db
                layer.uw = self.beta2 * layer.uw + (1-self.beta2) * np.square(layer.dw)
                layer.ub = self.beta2 * layer.ub + (1-self.beta2) * np.square(layer.db)

                mw_hat = layer.mw / (1-np.power(self.beta1, epoch))
                mb_hat = layer.mb / (1-np.power(self.beta1, epoch))
                uw_hat = layer.uw / (1-np.power(self.beta2, epoch))
                ub_hat = layer.ub / (1-np.power(self.beta2, epoch))

                layer.weight -= (self.learning_rate / (np.sqrt(uw_hat) + self.epsilon)) * ((self.beta1 * mw_hat) + (((1-self.beta1) * layer.dw) / (1-np.power(self.beta1,epoch))))
                layer.bias -= (self.learning_rate / (np.sqrt(ub_hat) + self.epsilon)) * ((self.beta1 * mb_hat) + (((1-self.beta1) * layer.db) / (1-np.power(self.beta1,epoch))))