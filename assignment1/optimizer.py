import numpy as np
from abc import ABC, abstractmethod

# abstract class for optimizers

class Optimizer(ABC):
    @abstractmethod
    def optimize(self):
        pass

class SGD(Optimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass

class MomentumSGD(Optimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass

class NestrovSGD(Optimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass

class RMSProp(Optimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass

class Adam(Optimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass

class NAdam(Optimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass