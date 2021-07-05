

from cnns import main
from dp_utils import get_noise_mul

MAX_GRAD_NORM = 0.1
MAX_EPS = 5

BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
BASE_LRS = [0.125, 0.25, 0.5, 1.0]
#BASE_LRS += [2.0, 4.0, 8.0]

TARGET_EPS = 3
TARGET_EPOCHS = [30, 60, 120]

for target_epoch in TARGET_EPOCHS:
    for base_lr in BASE_LRS:
        for bs in BATCH_SIZES:
            lr = (bs // 512) * base_lr
            mul = get_noise_mul(50000, bs, TARGET_EPS, target_epoch)

            print(f"epoch={target_epoch}, bs={bs}, lr={base_lr}*{bs//512}={lr}, mul={mul}")

            logdir = f"logs/cnns/cifar10/bs={bs}_lr={lr}_mul={mul:.2f}"
            main(dataset="cifar10", max_grad_norm=MAX_GRAD_NORM,
                 lr=lr, batch_size=bs, noise_multiplier=mul,
                 max_epsilon=MAX_EPS, logdir=logdir, epochs=150)
