import argparse
import os
import numpy as np

import torch
import torch.nn as nn
from opacus import PrivacyEngine

from train_utils import get_device, train, test
from data import get_data, SemiSupervisedSampler, get_scatter_transform, \
    get_scattered_loader, get_scattered_dataset
from models import CNNS, get_num_params, ScatterLinear
from dp_utils import ORDERS, get_privacy_spent, get_renyi_divergence, scatter_normalization
from log import Logger


def main(tiny_images=None, model="cnn", augment=False, use_scattering=False,
         batch_size=2048, mini_batch_size=256, lr=1, lr_start=None, optim="SGD",
         momentum=0.9, noise_multiplier=1, max_grad_norm=0.1,
         epochs=100, bn_noise_multiplier=None, max_epsilon=None,
         data_size=550000, delta=1e-6, logdir=None):
    logger = Logger(logdir)

    device = get_device()

    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size

    train_data, test_data = get_data("cifar10", augment=augment)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    if isinstance(tiny_images, torch.utils.data.Dataset):
        train_data_aug = tiny_images
    else:
        print("loading tiny images...")
        train_data_aug, _ = get_data("cifar10_500K", augment=augment,
                                     aux_data_filename=tiny_images)

    scattering, K, (h, w) = None, None, (None, None)
    pre_scattered = False
    if use_scattering:
        scattering, K, (h, w) = get_scatter_transform("cifar10_500K")
        scattering.to(device)

    # if the whole data fits in memory, pre-compute the scattering
    if use_scattering and data_size <= 50000:
        loader = torch.utils.data.DataLoader(train_data_aug, batch_size=100, shuffle=False, num_workers=4)
        train_data_aug = get_scattered_dataset(loader, scattering, device, data_size)
        pre_scattered = True

    assert data_size <= len(train_data_aug)
    num_sup = min(data_size, 50000)
    num_batches = int(np.ceil(50000 / mini_batch_size)) # cifar-10 equivalent

    train_batch_sampler = SemiSupervisedSampler(data_size, num_batches, mini_batch_size)
    train_loader_aug = torch.utils.data.DataLoader(train_data_aug,
                                                   batch_sampler=train_batch_sampler,
                                                   num_workers=0 if pre_scattered else 4,
                                                   pin_memory=not pre_scattered)

    rdp_norm = 0
    if model == "cnn":
        if use_scattering:
            save_dir = f"bn_stats/cifar10_500K"
            os.makedirs(save_dir, exist_ok=True)
            bn_stats, rdp_norm = scatter_normalization(train_loader,
                                                       scattering,
                                                       K,
                                                       device,
                                                       data_size,
                                                       num_sup,
                                                       noise_multiplier=bn_noise_multiplier,
                                                       orders=ORDERS,
                                                       save_dir=save_dir)
            model = CNNS["cifar10"](K, input_norm="BN", bn_stats=bn_stats)
            model = model.to(device)

            if not pre_scattered:
                model = nn.Sequential(scattering, model)
        else:
            model = CNNS["cifar10"](in_channels=3, internal_norm=False)

    elif model == "linear":
        save_dir = f"bn_stats/cifar10_500K"
        os.makedirs(save_dir, exist_ok=True)
        bn_stats, rdp_norm = scatter_normalization(train_loader,
                                                   scattering,
                                                   K,
                                                   device,
                                                   data_size,
                                                   num_sup,
                                                   noise_multiplier=bn_noise_multiplier,
                                                   orders=ORDERS,
                                                   save_dir=save_dir)
        model = ScatterLinear(K, (h, w), input_norm="BN", bn_stats=bn_stats)
        model = model.to(device)

        if not pre_scattered:
            model = nn.Sequential(scattering, model)
    else:
        raise ValueError(f"Unknown model {model}")
    model.to(device)

    if pre_scattered:
        test_loader = get_scattered_loader(test_loader, scattering, device)

    print(f"model has {get_num_params(model)} parameters")

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine(
        model,
        sample_rate=bs / data_size,
        alphas=ORDERS,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    privacy_engine.attach(optimizer)

    best_acc = 0
    flat_count = 0

    for epoch in range(0, epochs):

        print(f"\nEpoch: {epoch} ({privacy_engine.steps} steps)")
        train_loss, train_acc = train(model, train_loader_aug, optimizer, n_acc_steps=n_acc_steps)
        test_loss, test_acc = test(model, test_loader)

        if noise_multiplier > 0:
            print(f"sample_rate={privacy_engine.sample_rate}, "
                  f"mul={privacy_engine.noise_multiplier}, "
                  f"steps={privacy_engine.steps}")
            rdp_sgd = get_renyi_divergence(
                privacy_engine.sample_rate, privacy_engine.noise_multiplier
            ) * privacy_engine.steps
            epsilon, _ = get_privacy_spent(rdp_norm + rdp_sgd, target_delta=delta)
            epsilon2, _ = get_privacy_spent(rdp_sgd, target_delta=delta)
            print(f"ε = {epsilon:.3f} (sgd only: ε = {epsilon2:.3f})")

            if max_epsilon is not None and epsilon >= max_epsilon:
                return
        else:
            epsilon = None

        logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc, epsilon)
        logger.log_scalar("epsilon/train", epsilon, epoch)
        logger.log_scalar("cifar10k_loss/train", train_loss, epoch)
        logger.log_scalar("cifar10k_acc/train", train_acc, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            flat_count = 0
        else:
            flat_count += 1
            if flat_count >= 20:
                print("plateau...")
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', action="store_true")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--optim', type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_start', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--noise_multiplier', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', choices=["cnn", "resnet", "linear"], default="cnn")
    parser.add_argument('--tiny_images', default="ti_500K_pseudo_labeled.pickle")
    parser.add_argument('--use_scattering', action="store_true")
    parser.add_argument('--bn_noise_multiplier', type=float, default=0)
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--data_size', type=int, default=550_000)
    parser.add_argument('--delta', type=float, default=1e-6)
    args = parser.parse_args()
    main(**vars(args))
