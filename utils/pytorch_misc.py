import os
import re
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import DataParallel

def time_batch(gen, reset_every=100):
    start = time.time()
    start_t = 0
    for i, item in enumerate(gen):
        time_per_batch = (time.time() - start) / (i + 1- start_t)
        yield time_per_batch, item
        if i % reset_every == 0:
            start = time.time()
            start_t = i

class Flattener(torch.nn.Module):
    def __init__(self):
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def pad_sequence(sequence, lengths):
    output = sequence.new_zeros(len(lengths), max(lengths), *sequence.shape[1:])
    start = 0
    for i,diff in enumerate(lengths):
        if diff > 0:
            output[i, :diff] = sequence[start:(start+diff)]
        start += diff
    return output

def extra_leading_dim_in_sequence(f, x, mask):
    return f(x.view(-1, *x.shape[2:]), mask.view(-1, mask.shape[2])).view(*x.shape[:3], -1)

def find_checkpoint(save_dir, epoch=None):
    have_checkpoint = (save_dir is not None and any("model_state_epoch_" in x for x in os.listdir(save_dir)))
    if not have_checkpoint:
        print("there is no checkpoint ! please train model")
        return None
    if not epoch:
        checkpoint_files = os.listdir(save_dir)
        found_epochs = [re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1) for x in model_checkpoints] # [0,1,2,3,4,...]
        int_epochs = []
        for epoch in found_epochs:
            pieces = epoch.split(".")
            if len(pieces) == 1:
                int_epochs.append([int(pieces[0]), 0])
            else:
                int_epochs.append([int(pieces[0], pieces[1])])
        last_epoch = sorted(int_epochs, reverse=True)[0]
        if last_epoch[1] ==0:
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = "{0}.{1}".format(last_epoch[0], last_epoch[1])
    else:
        epoch_to_load = epoch
    model_path = os.path.join(save_dir, "model_state_epoch_{}.th".format(epoch_to_load))

    training_state_path = os.path.join(save_dir,"training_state_epoch_{}.th".format(epoch_to_load))
    return model_path, training_state_path

def save_checkpoint(model, optimizer, save_dir, epoch, val_metric_per_epoch, is_best=None, learning_rate_scheduler=None):
    if save_dir is not None:
        model_path = os.path.join(save_dir, "model_state_epoch_{}.th".format(epoch))
        model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
        torch.save(model_state, model_path)

        training_state = {"epoch":epoch,
                          "val_metric_per_epoch": val_metric_per_epoch,
                          "optimizer":optimizer.state_dict()}
        if learning_rate_scheduler is not None:
            training_state["learning_rate_schedule"] = learning_rate_scheduler.lr_scheduler.state_dict()
        training_path = os.path.join(save_dir, "training_state_epoch_{}.th".format(epoch))
        torch.save(training_state, training_path)

    if is_best:
        print("Best validation performance so fat. Copying weights to '{}/bert.th'.".format(save_dir))
        shutil.copyfile(model_path, os.path.join(save_dir, "best.th"))

def restore_best_checkpoint(model, save_dir):
    fn = os.path.join(save_dir, "best.th")
    model_state = torch.load(fn, map_location=device_mapping(-1))
    assert os.path.exists(fn)
    if isinstance(model, DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

def restore_checkpoint(model, optimizer, save_dir, learning_rate_scheduler=None, epoch=None):
    if epoch is None:
        checkpoint = find_checkpoint(save_dir)
    else:
        checkpoint = find_checkpoint(save_dir, epoch)

    if checkpoint is None:
        return 0, []

    model_path, training_state_path = checkpoint

    model_state = torch.load(model_path, map_location=device_mapping(-1))
    training_state = torch.load(training_state_path, map_location=device_mapping(-1))

    if isinstance(model, DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    optimizer.load_state_dict(training_state["optimizer"])

    if learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
        learning_rate_scheduler.lr_scheduler.load_state_dict(training_state["learning_rate_schedule"])
    move_optimizer_to_cuda(optimizer)

    if "val_metric_per_epoch" not in training_state:
        print("trainer state `val_metric_per_epoch` not found, using empty list")
        val_metric_per_epoch: []
    else:
        val_metric_per_epoch = training_state["val_metric_per_epoch"]

    if isinstance(training_state["epoch"], int):
        epoch_to_return = training_state["epoch"] + 1
    else:
        epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1
    return epoch_to_return, val_metric_per_epoch


def clip_grad_norm(named_parameters, max_norm, clip=True, verbose=False):
    """Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)
    parameters = [(n, p) for n, p in named_parameters if p.grad is not None]
    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
        param_to_norm[n] = param_norm
        param_to_shape[n] = tuple(p.size())
        if np.isnan(param_norm.item()):
            raise ValueError("the param {} was null.".format(n))

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef.item() < 1 and clip:
        for n, p in parameters:
            p.grad.data.mul_(clip_coef)

    if verbose:
        print('---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<60s}: {:.3f}, ({}: {})".format(name, norm, np.prod(param_to_shape[name]), param_to_shape[name]))
        print('-------------------------------', flush=True)

    return pd.Series({name: norm.item() for name, norm in param_to_norm.items()})











