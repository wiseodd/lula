#!/bin/bash

# MAP and baselines
for dataset in MNIST SVHN CIFAR-10 CIFAR-100; do
    python train.py --dataset $dataset

    if [[ $dataset = MNIST ]]; then
        python train.py --dataset $dataset --ood_loss --lenet
    else
        python train.py --dataset $dataset --ood_loss
    fi

    python train_baselines.py --dataset $dataset

    # LULAs
    python train_lula.py --dataset $dataset --base plain

    if [[ $dataset = MNIST ]]; then
        python train_lula.py --dataset $dataset --base oe --lenet
    else
        python train_lula.py --dataset $dataset --base plain
    fi
done
