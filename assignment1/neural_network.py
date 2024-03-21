import numpy as np
from helper import one_hot_encode, DataLoader
from layer import Linear
from activation import Sigmoid, Softmax, Tanh, ReLU
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
        elif self.activation == "tanh":
            self.activation_func = Tanh()
        elif self.activation == "relu":
            self.activation_func = ReLU()
    
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

    def sum_of_squared_weights(self):
        sum = 0.0
        for layer in self.layers:
            if isinstance(layer, Linear):
                sum += np.sum(np.square(layer.weight))
        return sum

    def forwardPass(self, xb):
        y_hat = xb
        for layer in self.layers:
            y_hat = layer(y_hat)
        return y_hat

    def backwardPass(self):
        prev_grad = self.loss_func.diff()
        for layer in self.layers[::-1]:
            prev_grad = layer.diff(prev_grad)

    def print_status(self):
        pass

    def run(self, xtrain, ytrain, xvalid, yvalid):
        n_batch = xtrain.shape[0] // self.batch_size
        data_loader = DataLoader(xtrain, ytrain, self.batch_size)
        for i in range(self.epochs):
            loss = []
            acc = []
            valid_loss = []
            valid_acc = []
            for _ in range(n_batch):
                
                # creating mini-batch
                xb, yb = data_loader()
                yb_enc = one_hot_encode(yb, self.out_dim)

                # forward prop
                y_hat = self.forwardPass(xb)

                loss.append(self.loss_func(yb_enc, y_hat) + ((self.weight_decay/2) * self.sum_of_squared_weights()))

                # calculate accuracy
                pred = np.argmax(y_hat, axis=-1)
                acc.append((pred==yb).sum() / self.batch_size)

                # backward prop
                self.backwardPass()
                
                # # gradient descent
                # for k, layer in enumerate(self.layers):
                #     if isinstance(layer, Linear):
                #         # print("layer: ", k, "dw norm: ", np.linalg.norm(layer.dw), "db norm: ", np.linalg.norm(layer.db))
                #         layer.weight -= self.learning_rate * (layer.dw + self.weight_decay * layer.weight) 
                #         layer.bias -= self.learning_rate * layer.db

                # momentum based gradient descent
                # for layer in self.layers:
                #     if isinstance(layer, Linear):
                #         layer.uw = self.beta * layer.uw + layer.dw
                #         layer.ub = self.beta * layer.ub + layer.db
                #         layer.weight -= self.learning_rate * layer.uw
                #         layer.bias -= self.learning_rate * layer.ub
                        
                # nesterov based gradient descent (to be implemented)
                

                # rmsprop based gradient descent
                # for layer in self.layers:
                #     if isinstance(layer, Linear):
                #         layer.uw = self.beta * layer.uw + (1-self.beta) * np.square(layer.dw)
                #         layer.ub = self.beta * layer.ub + (1-self.beta) * np.square(layer.db)
                #         layer.weight -= (self.learning_rate * layer.dw) / (np.sqrt(layer.uw) + self.epsilon)
                #         layer.bias -= (self.learning_rate * layer.db) / (np.sqrt(layer.ub) + self.epsilon)

                # adam based gradient descent
                for layer in self.layers:
                    if isinstance(layer, Linear):
                        layer.mw = self.beta1 * layer.mw + (1-self.beta1) * layer.dw
                        layer.mb = self.beta1 * layer.mb + (1-self.beta1) * layer.db
                        layer.uw = self.beta2 * layer.uw + (1-self.beta2) * np.square(layer.dw)
                        layer.ub = self.beta2 * layer.ub + (1-self.beta2) * np.square(layer.db)

                        # bias correcting
                        mw_hat = layer.mw / (1-np.power(self.beta1, i+1))
                        mb_hat = layer.mb / (1-np.power(self.beta1, i+1))
                        uw_hat = layer.uw / (1-np.power(self.beta2, i+1))
                        ub_hat = layer.ub / (1-np.power(self.beta2, i+1))

                        layer.weight -= (self.learning_rate * mw_hat) / (np.sqrt(uw_hat) + self.epsilon)
                        layer.bias -= (self.learning_rate * mb_hat) / (np.sqrt(ub_hat) + self.epsilon)

                # nadam based gradient descent (to be implemented)
                

            # valid forward prop
            n_batch = xvalid.shape[0] // self.batch_size

            for k in range(n_batch):
                start = k * self.batch_size
                end = start + self.batch_size
                xb = xvalid[start:end, :]
                yb = yvalid[start:end]
                yb_enc = one_hot_encode(yb, self.out_dim)

                y_hat = self.forwardPass(xb)

                valid_loss.append(self.loss_func(yb_enc, y_hat) + ((self.weight_decay/2) * self.sum_of_squared_weights()))

                # calculate accuracy
                pred = np.argmax(y_hat, axis=-1)
                valid_acc.append((pred==yb).sum() / self.batch_size)

            # print stats per epoch
            print("- - - - - - - - - - - -")
            print(f'epoch {i}')
            print(f"loss: {np.array(loss).mean()}")
            print(f"acc: {np.array(acc).mean()}")
            print(f"valid loss: {np.array(valid_loss).mean()}")
            print(f"valid acc: {np.array(valid_acc).mean()}")




    