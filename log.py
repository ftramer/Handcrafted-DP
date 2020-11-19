import numpy as np
import os
import shutil
import sys
from torch.utils.tensorboard import SummaryWriter
import torch


def model_input(data, device):
    datum = data.data[0:1]
    if isinstance(datum, np.ndarray):
        return torch.from_numpy(datum).float().to(device)
    else:
        return datum.float().to(device)


def get_script():
    py_script = os.path.basename(sys.argv[0])
    return os.path.splitext(py_script)[0]


def get_specified_params(hparams):
    keys = [k.split("=")[0][2:] for k in sys.argv[1:]]
    specified = {k: hparams[k] for k in keys}
    return specified


def make_hparam_str(hparams, exclude):
    return ",".join([f"{key}_{value}"
                     for key, value in sorted(hparams.items())
                     if key not in exclude])


class Logger(object):
    def __init__(self, logdir):

        if logdir is None:
            self.writer = None
        else:
            if os.path.exists(logdir) and os.path.isdir(logdir):
                shutil.rmtree(logdir)

            self.writer = SummaryWriter(log_dir=logdir)

    def log_model(self, model, input_to_model):
        if self.writer is None:
            return
        self.writer.add_graph(model, input_to_model)

    def log_epoch(self, epoch, train_loss, train_acc, test_loss, test_acc, epsilon=None):
        if self.writer is None:
            return
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/test", test_loss, epoch)
        self.writer.add_scalar("Accuracy/train", train_acc, epoch)
        self.writer.add_scalar("Accuracy/test", test_acc, epoch)

        if epsilon is not None:
            self.writer.add_scalar("Acc@Eps/train", train_acc, 100*epsilon)
            self.writer.add_scalar("Acc@Eps/test", test_acc, 100*epsilon)

    def log_scalar(self, tag, scalar_value, global_step):
        if self.writer is None or scalar_value is None:
            return
        self.writer.add_scalar(tag, scalar_value, global_step)
