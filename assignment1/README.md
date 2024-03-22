## Assignment 1

`run.sh`is wrapper file for `train.py`. Arguments can be changed inside `run.sh`. After changing arguments, you can simply execute it.
```
./run.sh
```
If using train.py directly, you will have to pass all the necessary arguments. \
To see usage:
```
python train.py --help
```
Following is the configuration that worked best on Fashion Mnist:
```
epochs=10
batch_size=32
loss=cross_entropy
optimizer=nadam
learning_rate=1e-3
momentum=0.9
beta=0.9
beta1=0.9
beta2=0.999
epsilon=1e-8
weight_decay=0.005
weight_init=xavier
num_layers=3
hidden_size=256
activation=tanh
```
