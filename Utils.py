import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from dataset.CTDataset import CTDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR


def load_data(config):
    dset_train = CTDataset(data_path=config.train_path, k_fold=False, train=True)
    dset_val = CTDataset(data_path=config.val_path, k_fold=False, train=False)

    train_loader = DataLoader(dset_train, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dset_val, batch_size=config.test_batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


def get_optimizer(args, net: nn.Module):
    lr, weight_decay = args.lr, args.weight_decay

    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr)
        print(f'<<<<<<<<<<<<<<<<<<<<AdamW>>>>>>>>>>>>>>>>>>>>')
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.8, nesterov=True)
        print(f'<<<<<<<<<<<<<<<<<<<<SGD>>>>>>>>>>>>>>>>>>>>')
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr)  # default is adam
        print(f'<<<<<<<<<<<<<<<<<<<<Adam>>>>>>>>>>>>>>>>>>>>')

    return optimizer


def get_scheduler(args, optimizer: torch.optim):
    epochs = args.epochs
    step_size = args.weight_decay

    if args.scheduler == 'warmup_cosine':
        warmup = args.warmup_epochs
        warmup_cosine_lr = (lambda epoch: epoch / warmup
        if epoch <= warmup else 0.5 * (
                math.cos((epoch - warmup) / (epochs - warmup) * math.pi) + 1))
        lr_scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)
    elif args.scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif args.scheduler == 'step':
        lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.scheduler == 'poly':
        lr_scheduler = LambdaLR(optimizer, lambda epoch: (1 - epoch / epochs) ** 0.9)
    elif args.scheduler == 'none':
        lr_scheduler = None
    else:
        raise NotImplementedError(f"LR scheduler {args.scheduler} is not implemented.")

    return lr_scheduler


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop
