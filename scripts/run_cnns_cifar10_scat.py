
from cnns import main
from dp_utils import get_noise_mul, get_renyi_divergence

MAX_GRAD_NORM = 0.1
MAX_EPS = 3.5

BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
BASE_LRS = [0.125, 0.25, 0.5, 1.0]

TARGET_EPS = 3
TARGET_EPOCHS = [30, 60, 120]

BN_MULS = [6, 8]
GN = [9, 27, 81]

for target_epoch in TARGET_EPOCHS:
    for base_lr in BASE_LRS:
        for bs in BATCH_SIZES:
            for bn_mul in BN_MULS:
                rdp_norm = 2 * get_renyi_divergence(1.0, bn_mul)
                mul = get_noise_mul(50000, bs, TARGET_EPS, target_epoch, rdp_init=rdp_norm)
                lr = (bs // 512) * base_lr
                print(f"epoch={target_epoch}, bs={bs}, lr={base_lr}*{bs//512}={lr}, mul={mul}, bn={bn_mul}")
                logdir = f"logs/cnns+scat/cifar10/bs={bs}_lr={lr}_mul={mul:.2f}_bn={bn_mul}"
                main(dataset="cifar10", max_grad_norm=MAX_GRAD_NORM,
                     lr=lr, batch_size=bs, noise_multiplier=mul,
                     use_scattering=True, input_norm="BN", bn_noise_multiplier=bn_mul,
                     max_epsilon=MAX_EPS, logdir=logdir, epochs=int(1.25*target_epoch))

for target_epoch in TARGET_EPOCHS:
    for base_lr in BASE_LRS:
        for bs in BATCH_SIZES:
            for group in GN:
                mul = get_noise_mul(50000, bs, TARGET_EPS, target_epoch, rdp_init=0)
                lr = (bs // 512) * base_lr
                print(f"epoch={target_epoch}, bs={bs}, lr={base_lr}*{bs//512}={lr}, mul={mul}, GN={group}")
                logdir = f"logs/cnns+scat/cifar10/bs={bs}_lr={lr}_mul={mul:.2f}_GN={group}"
                main(dataset="cifar10", max_grad_norm=MAX_GRAD_NORM,
                     lr=lr, batch_size=bs, noise_multiplier=mul,
                     use_scattering=True, input_norm="GroupNorm", num_groups=group,
                     max_epsilon=MAX_EPS, logdir=logdir, epochs=int(1.25*target_epoch))
