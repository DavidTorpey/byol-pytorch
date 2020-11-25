import pathlib
import sys

import torch
import yaml
from torchvision import datasets

import numpy as np
from .models.projection_head import MLPHead
from .models.backbone import ResNet18
from .trainer import BYOLTrainer
from .data.dataset import CustomDataset
from .data.utils import get_train_validation_data_loaders

torch.manual_seed(0)


def main():
    run_folder = sys.argv[-2]

    path = pathlib.Path('./results/{}/checkpoints'.format(run_folder))
    path.mkdir(exist_ok=True, parents=True)

    config = yaml.load(open(sys.argv[-1], "r"), Loader=yaml.FullLoader)
    dataset = config['data']['dataset']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
    print('Using dataset:', dataset)

    if dataset == 'stl10':
        train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True)
        trainset = train_dataset.data
        trainset = np.swapaxes(np.swapaxes(trainset, 1, 2), 2, 3)
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, download=True)
        trainset = train_dataset.data
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN('./data', split='train', download=True)
        trainset = train_dataset.data
        trainset = np.swapaxes(np.swapaxes(trainset, 1, 2), 2, 3)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100('./data', train=True, download=True)
        trainset = train_dataset.data

    train_dataset = CustomDataset(trainset, config['data'])

    # online network
    online_network = ResNet18(**config['network']).to(device)

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['mlp_head']).to(device)

    # target encoder
    target_network = ResNet18(**config['network']).to(device)

    optimizer = torch.optim.SGD(
        list(online_network.parameters()) + list(predictor.parameters()),
        **config['optimizer']
    )

    batch_size = config['trainer']['batch_size']
    epochs = config['trainer']['epochs']
    K = int(len(train_dataset) / batch_size) * config['trainer']['epochs']
    m = config['trainer']['m']

    train_loader, valid_loader = get_train_validation_data_loaders(
        train_dataset, 0.05, batch_size,
        config['trainer']['num_workers']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    trainer = BYOLTrainer(
        online_network=online_network,
        target_network=target_network,
        predictor=predictor,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        dataset=dataset,
        m=m,
        run_folder=run_folder
    )

    trainer.train(train_loader, valid_loader, K)


if __name__ == '__main__':
    main()
