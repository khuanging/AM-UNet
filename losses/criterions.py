import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from monai.metrics import HausdorffDistanceMetric


def expand_target(x, n_class, mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 2)
        xx[:, 3, :, :, :] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 1)
        xx[:, 1, :, :, :] = (x == 2)
        xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)


def dice_score(input, target, eps=1e-8):
    target = target.float()
    num = 2 * (input * target).sum()
    den = input.sum() + target.sum() + eps
    return num / den


def softmax_output_dice(output, target):
    ret = []
    output = F.softmax(output, dim=1).argmax(1)
    # whole WT
    o = output > 0;
    t = target > 0  # ce
    ret += dice_score(o, t),
    # core TC
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # active ET
    o = (output == 3);
    t = (target == 4)
    ret += dice_score(o, t),
    print(f'ret:{ret}')

    return ret


def Dice(input, target, eps=1e-5, weight=None):
    target = target.float()
    num = 2 * (input * target).sum()
    den = input.sum() + target.sum() + eps
    return 1.0 - num / den


def softmax_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    return loss1 + loss2 + loss3, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data


def softmax_dice2(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss0 = Dice(output[:, 0, ...], (target == 0).float())
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())
    # print(f'loss0:{loss0}, loss1:{loss1}, loss2:{loss2}, loss3:{loss3}')

    return loss1 + loss2 + loss3 + loss0, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data


def sigmoid_dice(output, target):
    '''
    The dice loss for using sigmoid activation function
    :param output: (b, num_class-1, d, h, w)
    :param target: (b, d, h, w)
    :return:
    '''
    print(f'------使用Sigmoid Dice--------')
    loss1 = Dice(output[:, 0, ...], (target == 1).float())
    loss2 = Dice(output[:, 1, ...], (target == 2).float())
    loss3 = Dice(output[:, 2, ...], (target == 4).float())

    return loss1 + loss2 + loss3, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data


def Generalized_dice(output, target, eps=1e-5, weight_type='square'):
    if target.dim() == 4:  # (b, h, w, d)
        target[target == 4] = 3  # transfer label 4 to 3
        target = expand_target(target, n_class=output.size()[1])  # extend target from (b, h, w, d) to (b, c, h, w, d)

    output = flatten(output)[1:, ...]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2 * intersect[0] / (denominator[0] + eps)
    loss2 = 2 * intersect[1] / (denominator[1] + eps)
    loss3 = 2 * intersect[2] / (denominator[2] + eps)

    return 1 - 2. * intersect_sum / denominator_sum, loss1, loss2, loss3


def Dual_focal_loss(output, target):
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    if target.dim() == 4:  # (b, h, w, d)
        target[target == 4] = 3  # transfer label 4 to 3
        target = expand_target(target, n_class=output.size()[1])  # extend target from (b, h, w, d) to (b, c, h, w, d)

    target = target.permute(1, 0, 2, 3, 4).contiguous()
    output = output.permute(1, 0, 2, 3, 4).contiguous()
    target = target.view(4, -1)
    output = output.view(4, -1)
    log = 1 - (target - output) ** 2

    return -(F.log_softmax((1 - (target - output) ** 2), 0)).mean(), 1 - loss1.data, 1 - loss2.data, 1 - loss3.data


def dice_coef_metric(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     treshold: float = 0.5,
                     eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Dice score for CC_MRI batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert (predictions.shape == truth.shape)
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def jaccard_coef_metric(probabilities: torch.Tensor,
                        truth: torch.Tensor,
                        treshold: float = 0.5,
                        eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Jaccard index for CC_MRI batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: jaccard score aka iou."
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert (predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def hausdorff_coef_metric(probabilities: torch.Tensor,
                          truth: torch.Tensor,
                          treshold: float = 0.5) -> np.ndarray:
    """
    Calculate Jaccard index for CC_MRI batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: jaccard score aka iou."
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert (predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i].unsqueeze(0)
        truth_ = truth[i].unsqueeze(0)

        hd95 = compute_hausdorff(prediction, truth_)
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(0.0)
        else:
            scores.append(np.array(hd95))
    return np.mean(scores)


class Meter:
    '''factory for storing and updating iou and dice scores.'''

    def __init__(self, treshold: float = 0.5):
        self.threshold: float = treshold
        self.dice_scores: list = []
        self.iou_scores: list = []
        self.hd95_scores: list = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Takes: logits from output model and targets,
        calculates dice and iou scores, and stores them in lists.
        """
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)
        hd95 = hausdorff_coef_metric(probs, targets, self.threshold)

        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.hd95_scores.append(hd95)

    def get_metrics(self) -> np.ndarray:
        """
        Returns: the average of the accumulated dice and iou scores.
        """
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        hd95 = np.nanmean(self.hd95_scores)
        return np.around(dice, 3), np.around(iou, 3), np.around(hd95, 3)


class DiceLoss(nn.Module):
    """Calculate dice loss."""

    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.contiguous().view(num, -1)
        targets = targets.contiguous().view(num, -1)
        assert (probability.shape == targets.shape)

        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        # print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """Compute objective loss: BCE loss + DICE loss."""

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert (logits.shape == targets.shape)
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)

        return bce_loss + dice_loss


class BCEDiceLossWeight(nn.Module):
    """Compute objective loss: BCE loss + DICE loss."""

    def __init__(self):
        super(BCEDiceLossWeight, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert (logits.shape == targets.shape)
        dice_loss = self.dice(logits, targets)
        dice_losses = dice_per_classes(logits, targets, classes=['D1', 'D2', 'D3'])
        bce_loss = self.bce(logits, targets)
        d1_dice = self.get_average_dice(dice_losses["D1"])
        d2_dice = self.get_average_dice(dice_losses["D2"])
        d3_dice = self.get_average_dice(dice_losses["D3"])
        dice_loss_weight = dice_loss * 0.5 + d1_dice * 0.2 + d2_dice * 0.1 + d3_dice * 0.2

        return bce_loss + dice_loss_weight

    def get_average_dice(self, data):
        dice_sum = 0.0
        for d in data:
            dice_sum += d
        return dice_sum / len(data)


def dice_per_classes(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     threshold: float = 0.5,
                     eps: float = 1e-9,
                     classes: list = ['WT', 'TC', 'ET']) -> torch.Tensor:
    """
    Calculate Dice score for CC_MRI batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with dice scores for each class.
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= threshold).float()
    assert (predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores


# helper functions for testing.
def dice_coef_metric_per_classes(probabilities: np.ndarray,
                                 truth: np.ndarray,
                                 treshold: float = 0.5,
                                 eps: float = 1e-9,
                                 classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """
    Calculate Dice score for CC_MRI batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with dice scores for each class.
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert (predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores


def jaccard_coef_metric_per_classes(probabilities: np.ndarray,
                                    truth: np.ndarray,
                                    treshold: float = 0.5,
                                    eps: float = 1e-9,
                                    classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """
    Calculate Jaccard index for CC_MRI batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        classes: list with name classes.
        Returns: dict with jaccard scores for each class."
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert (predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = (prediction * truth_).sum()
            union = (prediction.sum() + truth_.sum()) - intersection + eps
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)

    return scores


def compute_hausdorff(predictions: np.ndarray,
                      truth: np.ndarray,
                      include_background: bool = False,
                      get_not_nans: bool = True,
                      percentile: int = 95, ) -> np.ndarray:
    """
    :param predictions: model outputs after activation function.
    :param truth: model targets.
    :param include_background:
    :param get_not_nans:
    :param percentile:
    :return:
    """
    assert (predictions.shape == truth.shape)
    Hausdorff = HausdorffDistanceMetric(include_background=include_background, percentile=percentile, reduction="mean",
                                        get_not_nans=get_not_nans)
    hd95 = Hausdorff(predictions, truth)

    return np.array(hd95)


def hausdorff_coef_metric_per_classes(probabilities: np.ndarray,
                                      truth: np.ndarray,
                                      treshold: float = 0.5,
                                      classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    """
    Calculate Jaccard index for CC_MRI batch and for each class.
    Params:
        probobilities: model outputs after activation function.
        truth: model targets.
        classes: list with name classes.
        Returns: dict with hausdorff scores for each class."
    """
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert (predictions.shape == truth.shape)

    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i, class_]
            prediction = prediction[np.newaxis, np.newaxis, :]
            prediction = torch.from_numpy(prediction)
            truth_ = truth[i, class_]
            truth_ = truth_[np.newaxis, np.newaxis, :]
            truth_ = torch.from_numpy(truth_)

            hd95 = compute_hausdorff(prediction, truth_)
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(0.0)
            else:
                scores[classes[class_]].append(np.array(hd95))

    return scores


def compute_scores_per_classes(logits,
                               targets,
                               classes: list = ['WT', 'TC', 'ET']):
    """
    Compute Dice and Jaccard coefficients for each class.
    Params:

    """
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}
    hd_scores_per_classes = {key: list() for key in classes}

    logits = logits.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    dice_scores = dice_coef_metric_per_classes(logits, targets, classes=classes)
    iou_scores = jaccard_coef_metric_per_classes(logits, targets, classes=classes)
    hd_scores = hausdorff_coef_metric_per_classes(logits, targets, classes=classes)

    for key in dice_scores.keys():
        dice_scores_per_classes[key].append(np.mean(dice_scores[key]))

    for key in iou_scores.keys():
        iou_scores_per_classes[key].append(np.mean(iou_scores[key]))

    for key in hd_scores.keys():
        hd_scores_per_classes[key].append(hd_scores[key])

    return dice_scores_per_classes, iou_scores_per_classes, hd_scores_per_classes
