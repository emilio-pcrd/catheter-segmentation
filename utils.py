"""
Code almost taken from:
- https://github.com/HiLab-git/ACELoss/blob/main/aceloss.py
- meta code for finetuning sam-vit models
"""

import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def convert_png_to_jpg(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            png_path = os.path.join(directory, filename)
            jpg_path = os.path.splitext(png_path)[0] + ".jpg"
            with Image.open(png_path) as img:
                img.save(jpg_path)
            os.remove(png_path)


class ACLoss(nn.Module):
    """
    Active Contour Loss
    based on sobel filter
    """

    def __init__(self, miu=1.0, classes=2):
        super(ACLoss, self).__init__()

        self.miu = miu
        self.classes = classes
        self.classes
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        self.sobel_x = nn.Parameter(
            torch.from_numpy(sobel_x).float().expand(self.classes, 1, 3, 3),
            requires_grad=False,
        )
        self.sobel_y = nn.Parameter(
            torch.from_numpy(sobel_y).float().expand(self.classes, 1, 3, 3),
            requires_grad=False,
        )

        self.diff_x = nn.Conv2d(
            self.classes,
            self.classes,
            groups=self.classes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.diff_x.weight = self.sobel_x
        self.diff_y = nn.Conv2d(
            self.classes,
            self.classes,
            groups=self.classes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.diff_y.weight = self.sobel_y

    def forward(self, prediction, label):
        device = prediction.device
        self.diff_x = self.diff_x.to(device)
        self.diff_y = self.diff_y.to(device)

        grd_x = self.diff_x(prediction)
        grd_y = self.diff_y(prediction)

        # length
        length = torch.sum(torch.abs(torch.sqrt(grd_x**2 + grd_y**2 + 1e-8)))
        length = (length - length.min()) / (length.max() - length.min() + 1e-8)
        length = torch.sum(length)

        # region
        label = label.float()
        c_in = torch.ones_like(prediction)
        c_out = torch.zeros_like(prediction)
        region_in = torch.abs(torch.sum(prediction * ((label - c_in) ** 2)))
        region_out = torch.abs(torch.sum((1 - prediction) * ((label - c_out) ** 2)))
        region = self.miu * region_in + region_out

        return region + length


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # Ensure that alpha is a tensor with weights for each class.
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
            else:
                self.alpha = torch.tensor(alpha).cuda()
        else:
            self.alpha = None

    def forward(self, y_pred, y_true):
        BCE_loss = nn.BCEWithLogitsLoss(reduction="none")(y_pred, y_true)

        if self.alpha is not None:
            # Apply weights for each class.
            alpha = self.alpha[y_true.data.view(-1).long()].view_as(y_true)
            BCE_loss = alpha * BCE_loss

        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean()


# Define Dice Loss
def dice_loss(y_pred, y_true):
    smooth = 1.0
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice


# Combine Dice Loss and Focal Loss
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, alpha=None):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = dice_loss
        self.focal_loss = FocalLoss(alpha=alpha)

    def forward(self, y_pred, y_true):
        dice_term = self.dice_weight * self.dice_loss(y_pred, y_true)
        focal_term = self.focal_weight * self.focal_loss(y_pred, y_true)
        combined_loss = dice_term + focal_term
        return combined_loss


def dice_metric(predicted_masks, ground_truth_masks, threshold=0.5, smooth=1.0):
    """
    Computes the Dice coefficient for evaluating segmentation performance.

    Args:
        predicted_masks (torch.Tensor): Logits output from the model (before thresholding).
        ground_truth_masks (torch.Tensor): Binary ground truth masks (0 or 1).
        threshold (float, optional): Threshold to binarize predictions. Default is 0.5.
        smooth (float, optional): Smoothing factor to avoid division by zero. Default is 1.0.

    Returns:
        torch.Tensor: Dice coefficient score (higher is better, range: 0 to 1).
    """
    predicted_binary = (torch.sigmoid(predicted_masks) > threshold).float()

    predicted_binary = predicted_binary.view(-1)
    ground_truth_masks = ground_truth_masks.view(-1)

    intersection = (predicted_binary * ground_truth_masks).sum()
    denominator = predicted_binary.sum() + ground_truth_masks.sum()

    dice_score = (2.0 * intersection + smooth) / (denominator + smooth)

    return dice_score
