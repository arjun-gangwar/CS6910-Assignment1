import numpy
import pandas
import argparse
from neural_network import NeuralNetwork

def main(args: argparse.Namespace):
    nn = NeuralNetwork(
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Parameters")
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
    main(args)