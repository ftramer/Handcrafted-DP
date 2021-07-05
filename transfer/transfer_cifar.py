import argparse
import numpy as np

import torch
import torch.nn as nn
from opacus import PrivacyEngine

from models import StandardizeLayer
from train_utils import get_device, train, test
from data import get_data
from dp_utils import ORDERS, get_privacy_spent, get_renyi_divergence
from log import Logger


def main(feature_path=None, batch_size=2048, mini_batch_size=256,
         lr=1, optim="SGD", momentum=0.9, nesterov=False, noise_multiplier=1,
         max_grad_norm=0.1, max_epsilon=None, epochs=100, logdir=None):

    logger = Logger(logdir)

    device = get_device()

    # get pre-computed features
    x_train = np.load(f"{feature_path}_train.npy")
    x_test = np.load(f"{feature_path}_test.npy")

    train_data, test_data = get_data("cifar10", augment=False)
    y_train = np.asarray(train_data.targets)
    y_test = np.asarray(test_data.targets)

    trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=False, num_workers=1, pin_memory=True)

    n_features = x_train.shape[-1]
    try:
        mean = np.load(f"{feature_path}_mean.npy")
        var = np.load(f"{feature_path}_var.npy")
    except FileNotFoundError:
        mean = np.zeros(n_features, dtype=np.float32)
        var = np.ones(n_features, dtype=np.float32)

    bn_stats = (torch.from_numpy(mean).to(device), torch.from_numpy(var).to(device))

    model = nn.Sequential(StandardizeLayer(bn_stats), nn.Linear(n_features, 10)).to(device)

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine(
        model,
        sample_rate=bs / len(train_data),
        alphas=ORDERS,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    privacy_engine.attach(optimizer)

    for epoch in range(0, epochs):
        print(f"\nEpoch: {epoch}")

        train_loss, train_acc = train(model, train_loader, optimizer, n_acc_steps=n_acc_steps)
        test_loss, test_acc = test(model, test_loader)

        if noise_multiplier > 0:
            rdp_sgd = get_renyi_divergence(
                privacy_engine.sample_rate, privacy_engine.noise_multiplier
            ) * privacy_engine.steps
            epsilon, _ = get_privacy_spent(rdp_sgd)
            print(f"ε = {epsilon:.3f}")

            if max_epsilon is not None and epsilon >= max_epsilon:
                return
        else:
            epsilon = None

        logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc, epsilon)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optim', type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action="store_true")
    parser.add_argument('--noise_multiplier', type=float, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--feature_path', default=None)
    parser.add_argument('--max_epsilon', type=float, default=None)
    parser.add_argument('--logdir', default=None)
    args = parser.parse_args()
    main(**vars(args))
