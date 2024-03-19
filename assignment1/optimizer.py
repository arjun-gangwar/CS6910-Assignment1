import numpy as np
from abc import ABC, abstractmethod

# abstract class for optimizers

class ABSOptimizer(ABC):
    @abstractmethod
    def optimize(self):
        pass

class SGD(ABSOptimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass

class MomentumSGD(ABSOptimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass

class NestrovSGD(ABSOptimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass

class RMSProp(ABSOptimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass

class Adam(ABSOptimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass

class NAdam(ABSOptimizer):
    def __init__(self):
        pass
    # overriding abstract method
    def optimize(self):
        pass