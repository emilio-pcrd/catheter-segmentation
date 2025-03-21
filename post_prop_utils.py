import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import cv2


def focal_dice_loss(preds, targets, alpha=0.5, gamma=2.0):
    targets_cls = targets.squeeze(1).long()
    ce_loss = F.cross_entropy(preds, targets_cls, reduction="none")
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma) * ce_loss
    focal_component = focal_loss.mean()

    # Dice Loss component
    preds_soft = F.softmax(preds, dim=1)
    targets_one_hot = F.one_hot(targets_cls, num_classes=3).permute(0, 3, 1, 2).float()

    intersection = (preds_soft * targets_one_hot).sum(dim=(2, 3))
    cardinality = preds_soft.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

    dice = 2 * intersection / (cardinality + 1e-6)
    dice_component = 1 - dice.mean()

    return alpha * focal_component + (1 - alpha) * dice_component


def compute_multiclass_iou(preds, targets, num_classes=2, ignore_bg=True):
    preds = torch.argmax(preds, dim=1)
    targets = targets.squeeze(1).long()

    class_ious = []

    for cls in range(num_classes):
        pred_mask = (preds == cls).float()
        target_mask = (targets == cls).float()

        intersection = (pred_mask * target_mask).sum((1, 2))
        union = ((pred_mask + target_mask) > 0).float().sum((1, 2))

        iou = (intersection + 1e-6) / (union + 1e-6)

        class_ious.append(iou.mean().item())

    if ignore_bg:
        mean_iou = sum(class_ious[1:]) / (num_classes - 1)
    else:
        mean_iou = sum(class_ious) / num_classes

    return mean_iou, class_ious


class PostProcessingNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=16):
        super(PostProcessingNet, self).__init__()

        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)

        self.bridge = self._conv_block(base_filters * 4, base_filters * 8)

        self.up1 = nn.ConvTranspose2d(
            base_filters * 8, base_filters * 4, kernel_size=2, stride=2
        )
        self.dec1 = self._conv_block(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(
            base_filters * 4, base_filters * 2, kernel_size=2, stride=2
        )
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)

        self.up3 = nn.ConvTranspose2d(
            base_filters * 2, base_filters, kernel_size=2, stride=2
        )
        self.dec3 = self._conv_block(base_filters * 2, base_filters)

        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        x = F.max_pool2d(enc1, kernel_size=2, stride=2)
        enc2 = self.enc2(x)
        x = F.max_pool2d(enc2, kernel_size=2, stride=2)
        enc3 = self.enc3(x)
        x = F.max_pool2d(enc3, kernel_size=2, stride=2)

        x = self.bridge(x)

        x = self.up1(x)
        x = F.pad(x, [0, enc3.size(3) - x.size(3), 0, enc3.size(2) - x.size(2)])
        x = torch.cat([x, enc3], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = F.pad(x, [0, enc2.size(3) - x.size(3), 0, enc2.size(2) - x.size(2)])
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = F.pad(x, [0, enc1.size(3) - x.size(3), 0, enc1.size(2) - x.size(2)])
        x = torch.cat([x, enc1], dim=1)
        x = self.dec3(x)

        x = self.final(x)
        return x


import numpy as np
import torch
from scipy.ndimage import label, generate_binary_structure


def pipeline_modifie(pred_mask, batch_idx=0, i=0):
    is_tensor = isinstance(pred_mask, torch.Tensor)
    original_device = None

    if is_tensor:
        original_device = pred_mask.device
        pred_mask_np = pred_mask.cpu().numpy()
    else:
        pred_mask_np = pred_mask

    input_had_channel_dim = False
    if pred_mask_np.ndim == 3 and pred_mask_np.shape[0] == 1:
        input_had_channel_dim = True
        pred_mask_np = pred_mask_np.squeeze(0)

    result_mask = pred_mask_np.copy()

    class2_mask = np.zeros_like(pred_mask_np, dtype=np.uint8)
    class2_mask[pred_mask_np == 2] = 1

    structure = generate_binary_structure(2, 2)
    labeled, num_features = label(class2_mask, structure=structure)

    if num_features == 0:
        if is_tensor:
            if input_had_channel_dim:
                result_mask = np.expand_dims(result_mask, 0)
            result_mask = torch.from_numpy(result_mask).to(original_device)
        return result_mask

    component_sizes = np.bincount(labeled.ravel())[1:]
    largest_comp_idx = np.argmax(component_sizes) + 1

    class2_processed = np.zeros_like(pred_mask_np, dtype=np.uint8)
    class2_processed[labeled == largest_comp_idx] = 2

    result_mask = np.where(pred_mask_np == 1, 1, 0)
    result_mask = np.where(
        class2_processed == 2, 2, result_mask
    )

    if is_tensor:
        if input_had_channel_dim:
            result_mask = np.expand_dims(result_mask, 0)
        result_mask = torch.from_numpy(result_mask).to(original_device)

    return result_mask



def compute_distance_map(mask):
    mask_numpy = mask.cpu().numpy()
    dist_map = torch.tensor(
        [distance_transform_edt(m) for m in mask_numpy],
        dtype=torch.float32,
        device=mask.device,
    )
    return dist_map


def boundary_loss(pred, target):
    pred = torch.sigmoid(pred)

    target_dist_map = compute_distance_map(target)
    loss = (pred * target_dist_map).mean()

    return loss


def postprocess_mask(mask):
    """
    Post-processes a binary mask using morphological operations and contour linking.
    """
    mask = mask.cpu().numpy().astype(np.uint8) * 255  # Convert to uint8
    kernel = np.ones((3, 3), np.uint8)

    # Morphological closing to connect broken parts
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and draw them to link broken catheter parts
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)

    cv2.drawContours(mask_filled, contours, -1, (255), thickness=cv2.FILLED)
    return torch.tensor(mask_filled / 255.0, dtype=torch.float32)
