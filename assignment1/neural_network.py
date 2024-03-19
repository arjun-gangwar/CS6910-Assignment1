import numpy as np
from helper import one_hot_encode
from layer import Linear
from activation import Sigmoid, Softmax
from loss import CrossEntropyLoss
from optimizer import SGD, MomentumSGD, NestrovSGD, RMSProp, Adam, NAdam

class NeuralNetwork():
    def __init__(self,
                 wandb_project: str,
                 wandb_entity: str,
                 dataset: str,
                 in_dim: int,
                 out_dim: int,
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
        # self.wandb_project = wandb_project
        # self.wandb_entity = wandb_entity
        # self.dataset = dataset
        self.in_dim = in_dim
        self.out_dim = out_dim
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
        self.output_func = None
        self.optimizer_func = None
        self.loss_func = None
        self.activation_func = None
        self.layers = []
        self.train_loss_history = []
        self.valid_loss_history = []
        self.train_acc_history = []
        self.valid_acc_history = []
        # initialize network
        self.init_activation()
        self.init_loss()
        self.init_network()

    def init_activation(self):
        if self.activation == "sigmoid":
            self.activation_func = Sigmoid()
    
    def init_loss(self):
        if self.loss == "cross_entropy":
            self.output_func = Softmax()
            self.loss_func = CrossEntropyLoss()
            
    def init_network(self):
        for i in range(self.num_layers):
            # Input Layer
            if i==0:
                self.layers.append(Linear(self.in_dim, self.hidden_size, self.weight_init))
                self.layers.append(self.activation_func)
            # Output Layer
            elif i == self.num_layers-1:
                self.layers.append(Linear(self.hidden_size, self.out_dim, self.weight_init))
                self.layers.append(self.output_func)
            # Hidden Layers
            else:
                self.layers.append(Linear(self.hidden_size, self.hidden_size, self.weight_init))
                self.layers.append(self.activation_func)

    def forwardPass(self):
        pass

    def backwardPass(self):
        pass

    def print_status(self):
        pass

    def run(self, xtrain, ytrain):
        ix = np.random.randint(0, xtrain.shape[0], (self.batch_size,))
        xb = xtrain[ix]
        yb = ytrain[ix]
        yb_enc = one_hot_encode(yb, self.out_dim)

        n_batch_per_epoch = xtrain.shape[0] // self.batch_size

        for i in range(self.epochs):
            loss = []
            acc = []
            for _ in range(n_batch_per_epoch):

                # forward prop
                y_hat = xb
                for layer in self.layers:
                    y_hat = layer(y_hat)

                loss.append(self.loss_func(yb_enc, y_hat))

                # calculate accuracy
                pred = np.argmax(y_hat, axis=-1)
                acc.append((pred==yb).sum() / self.batch_size)

                # backward prop
                prev_grad = self.loss_func.diff()
                for layer in self.layers[::-1]:
                    prev_grad = layer.diff(prev_grad)
                
                # gradient descent
                for layer in self.layers:
                    if isinstance(layer, Linear):
                        layer.weight -= self.learning_rate * layer.dw
                        layer.bias -= self.learning_rate * layer.db

            # print stats per epoch
            print("- - - - - - - - - - - -")
            print(f'epoch {i}')
            print(f"loss: {np.array(loss).mean()}")
            print(f"acc: {np.array(acc).mean()}")




    