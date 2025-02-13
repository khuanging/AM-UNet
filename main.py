import os

import torch
import wandb
import argparse
from datetime import datetime
from ptflops import get_model_complexity_info
from thop import profile

from train import Trainer
from Utils import load_data
from models.Mamba.AMUnet import AMUnet
from losses.criterions import BCEDiceLoss


def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # Dataset paths
    parser.add_argument('--train_path', type=str, help="Path to training dataset")
    parser.add_argument('--val_path', type=str, help="Path to validation dataset")
    parser.add_argument('--save_path', type=str, help="Path to save model weights")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--test_batch_size', type=int, default=4, help="Batch size for testing")
    parser.add_argument('--epochs', type=int, default=150, help="Number of epochs to train")
    parser.add_argument('--optim', type=str, default="adamw", choices=["adamw", "sgd"],
                        help="Optimizer to use (adamw or sgd)")
    parser.add_argument('--scheduler', type=str, default="step", help="Learning rate scheduler to use")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=30, help="Weight decay for optimizer")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early stopping")
    parser.add_argument('--n_classes', type=int, default=4, help="Number of classes")
    parser.add_argument('--loss_gamma', type=float, default=0.3, help="Gamma value for loss function")

    # Miscellaneous parameters
    parser.add_argument('--seed', type=int, default=3407, help="Random seed")

    # Model parameters
    parser.add_argument('--basemodel', type=str, default=basemodel, help="Base model type")
    parser.add_argument('--model_type', type=str, default="mamba", choices=["resnet", "mobile", "other", "mamba"],
                        help="Model type (resnet, mobile, other, mamba)")
    parser.add_argument('--model_depth', type=int, nargs='+', default=[2, 2, 2, 2], help="Model depth")
    parser.add_argument('--common_stride', type=int, default=4, help="Common stride for model")
    parser.add_argument('--transformer_dropout', type=float, default=0.2, help="Dropout rate for transformer")
    parser.add_argument('--transformer_nheads', type=int, default=6, help="Number of heads for transformer")
    parser.add_argument('--transformer_dim_feedforward', type=int, default=768,
                        help="Feedforward dimension for transformer")
    parser.add_argument('--transformer_enc_layers', type=int, default=6,
                        help="Number of encoder layers for transformer")
    parser.add_argument('--conv_dim', type=int, default=384, help="Convolutional dimension")
    parser.add_argument('--mask_dim', type=int, default=384, help="Mask dimension")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    basemodel = "AM-UNet"
    opt = get_args()
    net = AMUnet(args=opt, in_channels=1, num_classes=opt.n_classes)
    train_loader, val_loader = load_data(opt)
    trainer = Trainer(args=opt,
                      net=net,
                      criterion=BCEDiceLoss(),
                      train_loader=train_loader,
                      val_loader=val_loader)
    trainer.run()
