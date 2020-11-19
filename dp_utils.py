import os
import math

import numpy as np
import torch
import opacus.privacy_analysis as tf_privacy

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def get_renyi_divergence(sample_rate, noise_multiplier, orders=ORDERS):
    rdp = torch.tensor(
        tf_privacy.compute_rdp(
            sample_rate, noise_multiplier, 1, orders
        )
    )
    return rdp


def get_privacy_spent(total_rdp, target_delta=1e-5, orders=ORDERS):
    return tf_privacy.get_privacy_spent(orders, total_rdp, target_delta)


def get_epsilon(sample_rate, mul, num_steps, target_delta=1e-5, orders=ORDERS, rdp_init=0):
    # compute the epsilon budget spent after `num_steps` with batch sampling rate
    # of `sample_rate` and a noise multiplier of `mul`

    rdp = rdp_init + get_renyi_divergence(sample_rate, mul, orders=orders) * num_steps
    eps, _ = get_privacy_spent(rdp, target_delta=target_delta, orders=orders)
    return eps


def get_noise_mul(num_samples, batch_size, target_epsilon, epochs, rdp_init=0, target_delta=1e-5, orders=ORDERS):
    # compute the noise multiplier that results in a privacy budget
    # of `target_epsilon` being spent after a given number of epochs of DP-SGD.

    mul_low = 100
    mul_high = 0.1

    num_steps = math.floor(num_samples // batch_size) * epochs
    sample_rate = batch_size / (1.0 * num_samples)

    eps_low = get_epsilon(sample_rate, mul_low, num_steps, target_delta, orders, rdp_init=rdp_init)
    eps_high = get_epsilon(sample_rate, mul_high, num_steps, target_delta, orders, rdp_init=rdp_init)

    assert eps_low < target_epsilon
    assert eps_high > target_epsilon

    while eps_high - eps_low > 0.01:
        mul_mid = (mul_high + mul_low) / 2
        eps_mid = get_epsilon(sample_rate, mul_mid, num_steps, target_delta, orders, rdp_init=rdp_init)

        if eps_mid <= target_epsilon:
            mul_low = mul_mid
            eps_low = eps_mid
        else:
            mul_high = mul_mid
            eps_high = eps_mid

    return mul_low


def get_noise_mul_privbyiter(num_samples, batch_size, target_epsilon, epochs, target_delta=1e-5):
    mul_low = 100
    mul_high = 0.1

    eps_low = priv_by_iter_guarantees(epochs, batch_size, num_samples, mul_low, target_delta, verbose=False)
    eps_high = priv_by_iter_guarantees(epochs, batch_size, num_samples, mul_high, target_delta, verbose=False)

    assert eps_low < target_epsilon
    assert eps_high > target_epsilon

    while eps_high - eps_low > 0.01:
        mul_mid = (mul_high + mul_low) / 2
        eps_mid = priv_by_iter_guarantees(epochs, batch_size, num_samples, mul_mid, target_delta, verbose=False)

        if eps_mid <= target_epsilon:
            mul_low = mul_mid
            eps_low = eps_mid
        else:
            mul_high = mul_mid
            eps_high = eps_mid

    return mul_low


def scatter_normalization(train_loader, scattering, K, device,
                          data_size, sample_size,
                          noise_multiplier=1.0, orders=ORDERS, save_dir=None):
    # privately compute the mean and variance of scatternet features to normalize
    # the data.

    rdp = 0
    epsilon_norm = np.inf
    if noise_multiplier > 0:
        # compute the RDP spent in this step
        sample_rate = sample_size / (1.0 * data_size)
        rdp = 2*get_renyi_divergence(sample_rate, noise_multiplier, orders)
        epsilon_norm, _ = get_privacy_spent(rdp)

    # try loading pre-computed stats
    use_scattering = scattering is not None
    assert use_scattering
    mean_path = os.path.join(save_dir, f"mean_bn_{sample_size}_{noise_multiplier}_{use_scattering}.npy")
    var_path = os.path.join(save_dir, f"var_bn_{sample_size}_{noise_multiplier}_{use_scattering}.npy")

    print(f"Using BN stats for {sample_size}/{data_size} samples")
    print(f"With noise_mul={noise_multiplier}, we get ε_norm = {epsilon_norm:.3f}")

    try:
        print(f"loading {mean_path}")
        mean = np.load(mean_path)
        var = np.load(var_path)
        print(mean.shape, var.shape)
    except OSError:

        # compute the scattering transform and the mean and squared mean of features
        scatters = []
        mean = 0
        sq_mean = 0
        count = 0
        for idx, (data, target) in enumerate(train_loader):
            with torch.no_grad():
                data = data.to(device)
                if scattering is not None:
                    data = scattering(data).view(-1, K, data.shape[2]//4, data.shape[3]//4)
                if noise_multiplier == 0:
                    data = data.reshape(len(data), K, -1).mean(-1)
                    mean += data.sum(0).cpu().numpy()
                    sq_mean += (data**2).sum(0).cpu().numpy()
                else:
                    scatters.append(data.cpu().numpy())

                count += len(data)
                if count >= sample_size:
                    break

        if noise_multiplier > 0:
            scatters = np.concatenate(scatters, axis=0)
            scatters = np.transpose(scatters, (0, 2, 3, 1))

            scatters = scatters[:sample_size]

            # s x K
            scatter_means = np.mean(scatters.reshape(len(scatters), -1, K), axis=1)
            norms = np.linalg.norm(scatter_means, axis=-1)

            # technically a small privacy leak, sue me...
            thresh_mean = np.quantile(norms, 0.5)
            scatter_means /= np.maximum(norms / thresh_mean, 1).reshape(-1, 1)
            mean = np.mean(scatter_means, axis=0)

            mean += np.random.normal(scale=thresh_mean * noise_multiplier,
                                     size=mean.shape) / sample_size

            # s x K
            scatter_sq_means = np.mean((scatters ** 2).reshape(len(scatters), -1, K),
                                       axis=1)
            norms = np.linalg.norm(scatter_sq_means, axis=-1)

            # technically a small privacy leak, sue me...
            thresh_var = np.quantile(norms, 0.5)
            print(f"thresh_mean={thresh_mean:.2f}, thresh_var={thresh_var:.2f}")
            scatter_sq_means /= np.maximum(norms / thresh_var, 1).reshape(-1, 1)
            sq_mean = np.mean(scatter_sq_means, axis=0)
            sq_mean += np.random.normal(scale=thresh_var * noise_multiplier,
                                        size=sq_mean.shape) / sample_size
            var = np.maximum(sq_mean - mean ** 2, 0)
        else:
            mean /= count
            sq_mean /= count
            var = np.maximum(sq_mean - mean ** 2, 0)

        if save_dir is not None:
            print(f"saving mean and var: {mean.shape} {var.shape}")
            np.save(mean_path, mean)
            np.save(var_path, var)

    mean = torch.from_numpy(mean).to(device)
    var = torch.from_numpy(var).to(device)

    return (mean, var), rdp


def priv_by_iter_guarantees(epochs, batch_size, samples, noise_multiplier, delta=1e-5, verbose=True):
    """Tabulating position-dependent privacy guarantees."""
    if noise_multiplier == 0:
        if verbose:
            print('No differential privacy (additive noise is 0).')
        return np.inf

    if verbose:
        print('In the conditions of Theorem 34 (https://arxiv.org/abs/1808.06651) '
              'the training procedure results in the following privacy guarantees.')
        print('Out of the total of {} samples:'.format(samples))

    steps_per_epoch = samples // batch_size
    orders = np.concatenate([np.linspace(2, 20, num=181), np.linspace(20, 100, num=81)])
    for p in (.5, .9, .99):
        steps = math.ceil(steps_per_epoch * p)  # Steps in the last epoch.
        coef = 2 * (noise_multiplier)**-2 * (
            # Accounting for privacy loss
            (epochs - 1) / steps_per_epoch +  # ... from all-but-last epochs
            1 / (steps_per_epoch - steps + 1))  # ... due to the last epoch
        # Using RDP accountant to compute eps. Doing computation analytically is
        # an option.
        rdp = [order * coef for order in orders]
        eps, _ = get_privacy_spent(rdp, delta, orders)
        if verbose:
            print('\t{:g}% enjoy at least ({:.2f}, {})-DP'.format(
                p * 100, eps, delta))

    return eps
