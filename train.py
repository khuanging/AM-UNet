import sys
import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from IPython.display import clear_output
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from Utils import get_scheduler, get_optimizer, EarlyStopping
from losses.criterions import Meter, compute_scores_per_classes, BCEDiceLossWeight


class Trainer:
    """
    A class to manage the training process of a neural network.

    Attributes:
        display_plot (bool): Flag to display training plots after each epoch.
        net (nn.Module): The neural network model to be trained.
        k_fold (bool): Flag indicating if k-fold cross-validation is used.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        early_stopping (EarlyStopping): Early stopping mechanism.
        basemodel (str): Name of the base model.
        num_epochs (int): Number of training epochs.
        save_path (str): Directory to save the trained model.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        phases (list): List containing 'train' and 'val' phases.
        dataloaders (dict): Dictionary containing DataLoader for training and validation.
        best_dice (float): Best Dice score achieved during training.
        losses (dict): Dictionary to store loss values for each phase.
        dice_scores (dict): Dictionary to store Dice scores for each phase.
        jaccard_scores (dict): Dictionary to store Jaccard scores for each phase.
        hd95_scores (dict): Dictionary to store Hausdorff95 scores for each phase.
    """

    def __init__(self, args, net: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 criterion: nn.Module, display_plot: bool = False, k_fold: bool = False):
        """
        Initializes the Trainer with the given parameters.

        Args:
            args: Command-line arguments or configuration.
            net (nn.Module): The neural network model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            criterion (nn.Module): The loss function.
            display_plot (bool, optional): Flag to display training plots after each epoch. Defaults to False.
            k_fold (bool, optional): Flag indicating if k-fold cross-validation is used. Defaults to False.
        """
        self.display_plot = display_plot
        self.net = net.cuda()
        self.net = nn.DataParallel(self.net)
        self.k_fold = k_fold
        self.criterion = criterion
        self.optimizer = get_optimizer(args=args, net=self.net)
        self.scheduler = get_scheduler(args=args, optimizer=self.optimizer)
        self.early_stopping = EarlyStopping(patience=args.patience)
        self.basemodel = args.basemodel
        self.num_epochs = args.epochs
        self.save_path = args.save_path
        self.scaler = GradScaler()

        self.phases = ["train", "val"]
        self.dataloaders = {"train": train_loader, "val": val_loader}

        self.best_dice = 0.0
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}
        self.hd95_scores = {phase: [] for phase in self.phases}

        seed_everything(args.seed, workers=True)

    def _compute_loss_and_outputs(self, images: torch.Tensor, targets: torch.Tensor):
        """
        Computes the loss and model outputs for a batch of data.

        Args:
            images (torch.Tensor): Input images.
            targets (torch.Tensor): Ground truth masks.

        Returns:
            tuple: Loss value and model outputs.
        """
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        """
        Executes one epoch of training or validation.

        Args:
            epoch (int): Current epoch number.
            phase (str): Phase of training ('train' or 'val').

        Returns:
            tuple: Metrics including loss, Dice score, IoU, and HD95 scores.
        """
        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        d1_dice = d2_dice = d3_dice = d4_dice = 0.0
        d1_hd95, d2_hd95, d3_hd95, d4_hd95 = [], [], [], []

        self.optimizer.zero_grad()
        data_bar = tqdm(dataloader, file=sys.stdout)

        for itr, (mr, data_batch) in enumerate(data_bar):
            images, targets = data_batch['image'].cuda(), data_batch['mask'].cuda()
            loss, logits = self._compute_loss_and_outputs(images, targets.float())

            if phase == "train":
                loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            dice_scores_per_classes, iou_scores_per_classes, hd95_scores_per_classes = compute_scores_per_classes(
                logits, targets, ['D1', 'D2', 'D3', 'D4']
            )

            d1_dice += dice_scores_per_classes["D1"][0]
            d2_dice += dice_scores_per_classes["D2"][0]
            d3_dice += dice_scores_per_classes["D3"][0]
            d4_dice += dice_scores_per_classes["D4"][0]

            d1_hd95.extend(self._process_hd95_scores(hd95_scores_per_classes["D1"]))
            d2_hd95.extend(self._process_hd95_scores(hd95_scores_per_classes["D2"]))
            d3_hd95.extend(self._process_hd95_scores(hd95_scores_per_classes["D3"]))
            d4_hd95.extend(self._process_hd95_scores(hd95_scores_per_classes["D4"]))

            running_loss += loss.item()
            meter.update(logits.detach().cpu(), targets.detach().cpu())
            data_bar.desc = f'{phase} epoch[{epoch + 1}/{self.num_epochs}]'

        h1_mean = np.nanmean(d1_hd95)
        h2_mean = np.nanmean(d2_hd95)
        h3_mean = np.nanmean(d3_hd95)
        h4_mean = np.nanmean(d4_hd95)

        epoch_loss = running_loss / total_batches
        epoch_dice, epoch_iou, epoch_hd = meter.get_metrics()

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)
        print(
            f'[epoch {epoch + 1}] {phase}| loss:{round(epoch_loss, 3)}, dice:{epoch_dice}, iou:{round(epoch_iou, 3)}, hd95:{round(epoch_hd, 3)} '
            f'd1_dice:{np.around(d1_dice / total_batches, 3)}, d2_dice:{np.around(d2_dice / total_batches, 3)}, '
            f'd3_dice:{np.around(d3_dice / total_batches, 3)}, d4_dice:{np.around(d4_dice / total_batches, 3)}')

        return epoch_loss, epoch_dice, epoch_iou, d1_dice / total_batches, d2_dice / total_batches, d3_dice / total_batches, d4_dice / total_batches, h1_mean, h2_mean, h3_mean, h4_mean

    def run(self):
        best_epoch = 0
        best_epoch_result = []
        for epoch in range(self.num_epochs):
            train_loss, train_dice, train_iou, train_d1, train_d2, train_d3, train_d4, train_h1, train_h2, train_h3, train_h4 = self._do_epoch(
                epoch, "train")
            with torch.no_grad():
                val_loss, val_dice, val_iou, val_d1, val_d2, val_d3, val_d4, val_h1, val_h2, val_h3, val_h4 = self._do_epoch(
                    epoch,
                    "val")
                self.scheduler.step()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            if self.display_plot:
                self._plot_train_history()
            if self.early_stopping.step(val_dice):
                print("Early stopping")
                break

            if val_dice > self.best_dice:
                # print(f"\n{'#' * 20}\nSaved new checkpoint\n{'#' * 20}\n")
                self.best_dice = val_dice
                best_epoch = epoch
                best_epoch_result = [val_dice, val_iou, val_d1, val_d2, val_d3, val_d4]
                self._save_train_history()

            print(f'[epoch {epoch + 1}] train_loss:{np.around(train_loss, 3)}, val_loss:{np.around(val_loss, 3)}, '
                  f'best_dice:{round(self.best_dice, 3)}, best_epoch:{best_epoch + 1}, lr:{lr}')

        if self.k_fold:
            return best_epoch_result

    def _process_hd95_scores(self, scores):
        """
        To process the hd95_scores_per_classes values into a one-dimensional list:
        Parameters: scores can be a float, a single-layer list, or a multi-layer nested list.
        Returns: A flattened list of numerical values.
        """
        if isinstance(scores, (float, int)):  # 单个数值
            return [scores]
        elif isinstance(scores, list):  # 列表情况
            flat_scores = []
            for item in scores:
                flat_scores.extend(self._process_hd95_scores(item))  # 递归展平
            return flat_scores
        elif isinstance(scores, np.ndarray):  # numpy 数组情况
            return self._process_hd95_scores(scores.tolist())  # 转为列表递归处理
        else:
            raise ValueError(f"Unsupported data type: {type(scores)}")

    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.jaccard_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['val'][-1]}
            """,

            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['val'][-1]} 
            """,

            f"""
            train jaccard score {self.jaccard_scores['train'][-1]}
            val jaccard score {self.jaccard_scores['val'][-1]}
            """,
        ]

        clear_output(True)
        with plt.style.context("seaborn-dark-palette"):
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            for i, ax in enumerate(axes):
                ax.plot(data[i]['val'], c=colors[0], label="val")
                ax.plot(data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")

            plt.tight_layout()
            plt.show()

    def load_predtrain_model(self,
                             state_path: str):
        self.net.load_state_dict(torch.load(state_path))
        print("Predtrain model loaded")

    def _save_train_history(self):
        """writing model weights and training logs to files."""
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        save_path = os.path.join(self.save_path, self.basemodel + "_best_epoch"".pth")
        torch.save(self.net.state_dict(), save_path)

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores, self.hd95_scores]
        log_names_ = ["_loss", "_dice", "_jaccard", "_hd95"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                for key in logs_[i]]
        log_names = [key + log_names_[i]
                     for i in list(range(len(logs_)))
                     for key in logs_[i]
                     ]
