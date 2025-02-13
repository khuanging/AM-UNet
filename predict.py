import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from dataset.CTDataset import CTDataset
from models.Mamba.AMUnet import AMUnet
from losses.criterions import (
    Meter,
    compute_scores_per_classes,
    dice_coef_metric,
    jaccard_coef_metric,
    hausdorff_coef_metric,
)


def calculate_metric(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    Calculate evaluation metrics: Dice coefficient, IoU, and Hausdorff distance.

    Args:
        logits (torch.Tensor): Model output before activation.
        targets (torch.Tensor): Ground truth masks.
        threshold (float): Threshold to binarize the logits.

    Returns:
        tuple: Dice coefficient, IoU, and Hausdorff distance.
    """
    probs = torch.sigmoid(logits)
    dice = dice_coef_metric(probs, targets, threshold)
    iou = jaccard_coef_metric(probs, targets, threshold)
    hd95 = hausdorff_coef_metric(probs, targets, threshold)
    return dice, iou, hd95


def load_test_data(args):
    """
    Load test dataset and create DataLoader.

    Args:
        args: Command line arguments containing test_path and batch_size.

    Returns:
        DataLoader: DataLoader for the test dataset.
    """
    dset_test = CTDataset(data_path=args.test_path, k_fold=False, train=False, test=True)
    test_loader = DataLoader(dset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return test_loader


def predict_all_cases(config, network, test_loader):
    """
    Perform predictions on all test cases and evaluate metrics.

    Args:
        config: Configuration object containing model and training parameters.
        network: The neural network model to use for predictions.
        test_loader: DataLoader for the test dataset.
    """
    seed_everything(config.seed, workers=True)
    meter = Meter()
    model = torch.nn.DataParallel(network).cuda()
    model.load_state_dict(torch.load(config.model_path), strict=False)
    model.eval()

    d1_hd95, d2_hd95, d3_hd95, d4_hd95 = [], [], [], []
    d1_dice, d2_dice, d3_dice, d4_dice = [], [], [], []
    dice_list, iou_list, hd95_list = [], [], []

    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for step, (ct, data_batch, mask_info) in enumerate(test_bar):
            test_img, test_mask = data_batch['image'].cuda(), data_batch['mask'].cuda()
            outputs = model(test_img)
            dice_scores_per_classes, iou_scores_per_classes, hd95_scores_per_classes = compute_scores_per_classes(
                outputs, test_mask, ['D1', 'D2', 'D3', 'D4']
            )

            # Process mask_info to extract relevant details
            output_list = [[t.tolist() for t in inner_list] for inner_list in mask_info]
            output_list = [[inner_value[0] for inner_value in outer_value] for outer_value in output_list]
            mask_info = [tuple(inner_list) for inner_list in output_list]

            dice, iou, hd95 = calculate_metric(outputs.detach().cpu(), test_mask.detach().cpu())
            d1_dice.append(dice_scores_per_classes["D1"][0])
            d2_dice.append(dice_scores_per_classes["D2"][0])
            d3_dice.append(dice_scores_per_classes["D3"][0])
            d4_dice.append(dice_scores_per_classes["D4"][0])
            d1_hd95.extend(hd95_scores_per_classes["D1"][0])
            d2_hd95.append(hd95_scores_per_classes["D2"][0])
            d3_hd95.append(hd95_scores_per_classes["D3"][0])
            d4_hd95.append(hd95_scores_per_classes["D4"][0])

            dice_list.append(dice)
            iou_list.append(iou)
            hd95_list.append(hd95)

            # Convert and save the segmentation output
            convert_to_single_channel_3d(
                outputs.detach().cpu().squeeze(dim=0).numpy(),
                mask_info,
                ct[0],
                config.save_path
            )
            meter.update(outputs.detach().cpu(), test_mask.detach().cpu())

        epoch_dice, epoch_iou, epoch_hd = meter.get_metrics()

    print(
        f'Test Dice: {round(epoch_dice, 3)}  Test IoU: {round(epoch_iou, 3)} '
        f'Test HD95: {round(epoch_hd, 3)}, D1: {np.mean(d1_dice)}, D2: {np.mean(d2_dice)}, '
        f'D3: {np.mean(d3_dice)}, D4: {np.mean(d4_dice)}'
    )


def load_nifti_volume(img_path, mask_path):
    """
    Load NIfTI image and mask volumes.

    Args:
        img_path (str): Path to the image file.
        mask_path (str): Path to the mask file.

    Returns:
        tuple: Numpy arrays of the image and mask, along with their metadata.
    """
    image = sitk.ReadImage(img_path)
    np_image = sitk.GetArrayFromImage(image)
    mask = sitk.ReadImage(mask_path)
    np_mask = sitk.GetArrayFromImage(mask)
    direction = mask.GetDirection()
    origin = mask.GetOrigin()
    spacing = mask.GetSpacing()
    return np_image, np_mask, (direction, origin, spacing)


def convert_to_single_channel_3d(segmentation, mask_info, ct, save_path, threshold=0.5):
    """
    Convert multi-channel segmentation output to a single-channel 3D volume and save as NIfTI.

    Args:
        segmentation (np.ndarray): Multi-channel segmentation output.
        mask_info (tuple): Metadata including direction, origin, and spacing.
        ct (str): Identifier for the current CT scan.
        save_path (str): Directory to save the converted segmentation.
        threshold (float): Threshold to binarize the segmentation output.
    """
    num_labels, height, width, depth = segmentation.shape
    direction, origin, spacing = mask_info

    # Initialize single-channel segmentation
    single_channel_segmentation = np.zeros((height, width, depth), dtype=np.uint8)

    # Process each channel
    for idx in range(num_labels):
        seg_channel = segmentation[idx, ::]
        seg_channel = np.where(seg_channel >= threshold, idx + 1, 0)
        single_channel_segmentation[seg_channel == idx + 1] = idx + 1

    sitk_image = sitk.GetImageFromArray(single_channel_segmentation)
    sitk_image.SetDirection(direction)
    sitk_image.SetOrigin(origin)
    sitk_image.SetSpacing(spacing)

    # save
    if not os.path.exists(os.path.join(save_path, ct)):
        os.makedirs(os.path.join(save_path, ct))
    sitk.WriteImage(sitk_image, os.path.join(save_path, f'{basemodel}.seg.nrrd'))


if __name__ == '__main__':
    basemodel = "AMUnet"  #
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str,
                        default="")
    parser.add_argument('--model_path', type=str,
                        default=f"")
    parser.add_argument('--save_path', type=str,
                        default="")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3047)
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--loss', type=str, default='softmax_dice')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss_gamma', type=float, default=0.3)
    parser.add_argument('--model_type', type=str, default="mamba", help="resnet, mobile")
    parser.add_argument('--model_depth', type=int, default=[2, 2, 2, 2], help="[1, 1, 3, 3, 3]")
    parser.add_argument('--common_stride', type=int, default=4)
    parser.add_argument('--transformer_dropout', type=float, default=0.2)
    parser.add_argument('--transformer_nheads', type=int, default=6)
    parser.add_argument('--transformer_dim_feedforward', type=int, default=768)
    parser.add_argument('--transformer_enc_layers', type=int, default=6)
    parser.add_argument('--conv_dim', type=int, default=384)
    parser.add_argument('--mask_dim', type=int, default=384)

    opt = parser.parse_args()
    net = AMUnet(args=opt, in_channels=1, num_classes=opt.n_classes)
    test_data_loader = load_test_data(args=opt)
    predict_all_cases(opt, net, test_data_loader)
