#!/bin/bash

for i in {1..5}
do
    python eval_OOD.py --randseed $((RANDOM)) --dataset MNIST --ood_dset imagenet
done

for i in {1..5}
do
    python eval_OOD.py --randseed $((RANDOM)) --dataset SVHN --ood_dset imagenet
done

for i in {1..5}
do
    python eval_OOD.py --randseed $((RANDOM)) --dataset CIFAR10 --ood_dset imagenet
done

for i in {1..5}
do
    python eval_OOD.py --randseed $((RANDOM)) --dataset CIFAR100 --ood_dset imagenet
done


python eval_CIFAR10C.py --ood_dset imagenet
python eval_MNISTC.py --ood_dset imagenet --transform rotation
python eval_MNISTC.py --ood_dset imagenet --transform translation
