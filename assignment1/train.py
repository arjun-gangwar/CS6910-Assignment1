import wandb
import argparse
import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from neural_network import NeuralNetwork

def wandb_sweep():
    with wandb.init() as run:
        config = wandb.config
        epochs = config.epochs
        num_layers = config.num_layers
        learning_rate = config.learning_rate
        hidden_size = config.hidden_size
        weight_decay = config.weight_decay
        batch_size = config.batch_size
        optimizer = config.optimizer
        weight_init = config.weight_init
        activation = config.activation
        loss = config.loss

        run_name=f"ac_{activation}_hl_{num_layers}_hs_{hidden_size}_bs_{batch_size}_op_{optimizer}_ep_{epochs}"
        wandb.run.name=run_name
        nn = NeuralNetwork(
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                in_dim=784,
                out_dim=10,
                epochs=epochs,
                batch_size=batch_size,
                loss=loss,
                optimizer=optimizer,
                learning_rate=learning_rate,
                momentum=args.momentum,
                beta=args.beta,
                beta1=args.beta1,
                beta2=args.beta2,
                epsilon=args.epsilon,
                weight_decay=weight_decay,
                weight_init=weight_init,
                num_layers=num_layers,
                hidden_size=hidden_size,
                activation=activation,
            )
        nn.run(xtrain, ytrain, xvalid, yvalid)

def main(args: argparse.Namespace):
    if args.use_wandb == "true":
        wandb.login()
        sweep_config = {
            'method': 'bayes',
            'name' : 'sweep cross entropy',
            'metric': {
                'name': 'avg_valid_acc',
                'goal': 'maximize'
            },
            'parameters': {
                'epochs': {
                    'values': [5,10,15]
                },
                'num_layers': {
                    'values': [3,4,5]
                },
                'learning_rate': {
                    'values': [1e-3, 1e-4]
                },'hidden_size':{
                    'values': [32,64,128,256]
                },
                'weight_decay': {
                    'values': [0,0.0005,0.005,0.5]
                },'batch_size':{
                    'values': [16,32,64,128,256]
                },'optimizer':{
                    'values': ['sgd','momentum','nag','rmsprop','adam','nadam']
                },'weight_init': {
                    'values': ['random','xavier']
                },'activation':{
                    'values': ['sigmoid','tanh','relu']
                },'loss':{
                    'values':['cross_entropy']
                }
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project)

    if args.use_wandb == "true":
        wandb.agent(sweep_id, function=wandb_sweep, count=200)
        wandb.finish()
    else:
        nn = NeuralNetwork(
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            in_dim=784,
            out_dim=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss=args.loss,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            beta=args.beta,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            weight_decay=args.weight_decay,
            weight_init=args.weight_init,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            activation=args.activation,
        )
        
        nn.run(xtrain, ytrain, xvalid, yvalid)
        ypred = nn.test(xtest, ytest)

        if args.use_wandb == "true":
            wandb.init(project=args.wandb_project)
            wandb.run.name="confusion_matrix"
            wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None, y_true=ytest, preds=ypred, class_names=labels)})
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Parameters")
    parser.add_argument("-uw", 
                        "--use_wandb", 
                        type=str,
                        default="false",
                        help="Use Weights and Biases: [true, false]")
    parser.add_argument("-wp", 
                        "--wandb_project", 
                        type=str, 
                        default="myprojectname",
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we",
                        "--wandb_entity", 
                        type=str,
                        default="myname",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d",
                        "--dataset", 
                        type=str,
                        default="fashion_mnist",
                        help="choices: ['mnist', 'fashion_mnist']")
    parser.add_argument("-e",
                        "--epochs", 
                        type=int,
                        default=1,
                        help="Number of epochs to train neural network.")
    parser.add_argument("-b",
                        "--batch_size", 
                        type=int,
                        default=4,
                        help="Batch size used to train neural network.")
    parser.add_argument("-l",
                        "--loss", 
                        type=str,
                        default="cross_entropy",
                        help="choices: ['mean_squared_error', 'cross_entropy']")
    parser.add_argument("-o",
                        "--optimizer", 
                        type=str,
                        default="sgd",
                        help="choices: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")
    parser.add_argument("-lr",
                        "--learning_rate", 
                        type=float,
                        default=0.1,
                        help="Learning rate used to optimize model parameters.")
    parser.add_argument("-m",
                        "--momentum", 
                        type=float,
                        default=0.5,
                        help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta",
                        "--beta", 
                        type=float,
                        default=0.5,
                        help="Beta used by rmsprop optimizer.")
    parser.add_argument("-beta1",
                        "--beta1", 
                        type=float,
                        default=0.5,
                        help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2",
                        "--beta2", 
                        type=float,
                        default=0.5,
                        help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps",
                        "--epsilon", 
                        type=float,
                        default=0.000001,
                        help="Epsilon used by optimizers.")
    parser.add_argument("-w_d",
                        "--weight_decay", 
                        type=float,
                        default=.0,
                        help="Weight decay used by optimizers.")
    parser.add_argument("-w_i",
                        "--weight_init", 
                        type=str,
                        default="random",
                        help="choices: ['random', 'Xavier']")
    parser.add_argument("-nhl",
                        "--num_layers", 
                        type=int,
                        default=1,
                        help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz",
                        "--hidden_size", 
                        type=int,
                        default=4,
                        help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a",
                        "--activation", 
                        type=str,
                        default="sigmoid",
                        help="choices: ['identity', 'sigmoid', 'tanh', 'ReLU']")
    args = parser.parse_args()

    if args.dataset == 'fashion_mnist':
        (xtrain, ytrain), (xtest, ytest) = fashion_mnist.load_data()
        labels = {0: "T-shirt/top",
          1: "Trouser",
          2: "Pullover",
          3: "Dress",
          4: "Coat",
          5: "Sandal",
          6: "Shirt",
          7: "Sneaker",
          8: "Bag",
          9: "Ankle boot"}
    elif args.dataset == 'mnist':
        (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
        labels = {0: "0",
          1: "1",
          2: "2",
          3: "3",
          4: "4",
          5: "5",
          6: "6",
          7: "7",
          8: "8",
          9: "9"}

    # split test into  valid and test
    idx = np.random.permutation(xtrain.shape[0])
    shuffled_xtrain = xtrain[idx,:,:]
    shuffled_ytrain = ytrain[idx]

    n = int(xtrain.shape[0] * 0.90)
    xtrain, ytrain = shuffled_xtrain[:n,:,:], shuffled_ytrain[:n]
    xvalid, yvalid = shuffled_xtrain[n:,:,:], shuffled_ytrain[n:]

    # normalizing data
    xtrain = xtrain.reshape(xtrain.shape[0], -1) / 255
    xvalid = xvalid.reshape(xvalid.shape[0], -1) / 255
    xtest = xtest.reshape(xtest.shape[0], -1) / 255

    print(f"{xtrain.shape=} {ytrain.shape=}")
    print(f"{xvalid.shape=} {yvalid.shape=}")
    print(f"{xtest.shape=} {ytest.shape=}")

    main(args)
