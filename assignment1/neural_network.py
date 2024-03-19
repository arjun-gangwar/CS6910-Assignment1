import numpy as np
from layer import Linear
from activation import Sigmoid, Softmax
from loss import CrossEntropyLoss
from optimizer import SGD, MomentumSGD, NestrovSGD, RMSProp, Adam, NAdam

class NeuralNetwork():
    def __init__(self,
                 wandb_project: str,
                 wandb_entity: str,
                #  dataset: str,
                 epochs: int,
                 batch_size: int,
                 loss: str,
                 optimizer: str,
                 learning_rate: float,
                 momentum: float,
                 beta: float,
                 beta1: float,
                 beta2: float,
                 epsilon: float,
                 weight_decay: float,
                 weight_init: str,
                 num_layers: int,
                 hidden_size: int,
                 activation: str):
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.weight_init = weight_init
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.output_activation = None
        self.train_loss_history = []
        self.valid_loss_history = []
        self.train_acc_history = []
        self.valid_acc_history = []

    def init_activation(self):
        pass
    
    def init_loss(self):
        pass

    def init_network(self):
        pass





    