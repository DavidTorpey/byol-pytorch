# byol-pytorch

PyTorch implementation of the [BYOL](https://arxiv.org/abs/2006.07733)
contrastive self-supervised learning model.

The implementation is influenced by [this repo](https://github.com/sthalles/PyTorch-BYOL).

## Running Self-Supervised Pre-training

Simply run the following:

```bash
python3 -m folder_name src.run config/config_cifar10.yaml
``` 

## Running Linear Evaluation

Simply run the following:

```bash
python3 -m folder_name src.linear_eval.run config/config_cifar10.yaml
```

## Things to Note

1. This implementation is focused on running BYOL for small-scale datasets
such as CIFAR10, CIFAR100, and SVHN - not ImageNet. Thus, we use the Adam
optimiser instead of the LARS optimiser, since this seems to perform better.
1. The linear evaluation protocol is different to the original work. We perform
no data augmentation for this phase.
