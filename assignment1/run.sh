#!/bin/bash
wandb_project=fake
wandb_entity=fake
dataset=fake
epochs=10
batch_size=64
loss=cross_entropy
optimizer=sgd
learning_rate=1e-4
momentum=0.5
beta=0.5
beta1=0.5
beta2=0.5
epsilon=0.000001
weight_decay=0
weight_init=random
num_layers=3
hidden_size=128
activation=sigmoid

python train.py \
    --wandb_project ${wandb_project} \
    --wandb_entity ${wandb_entity} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --loss ${loss} \
    --optimizer ${optimizer} \
    --learning_rate ${learning_rate} \
    --momentum ${momentum} \
    --beta ${beta} \
    --beta1 ${beta1} \
    --beta2 ${beta2} \
    --epsilon ${epsilon} \
    --weight_decay ${weight_decay} \
    --weight_init ${weight_init} \
    --num_layers ${num_layers} \
    --hidden_size ${hidden_size} \
    --activation ${activation}