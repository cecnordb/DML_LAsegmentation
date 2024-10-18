import torch.nn as nn
import torch

def dice_coefficient(pred, target, smooth=1.):
    pred = pred > 0.5  # Thresholding to get binary predictions
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def soft_dice_coefficient(pred, target, smooth=1.):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def soft_dice_loss(pred, target, smooth=1.):
    return 1 - soft_dice_coefficient(pred, target, smooth)

def soft_iou(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def iou(pred, target, smooth=1e-6):
    pred = pred > 0.5  # Thresholding to get binary predictions
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

class CombinedLoss(nn.Module):
    """
    Expects logit inputs
    """

    def __init__(self, smooth=1):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, outputs, targets):
        ce_loss = self.bce_loss(outputs, targets)
        dice = soft_dice_loss(torch.sigmoid(outputs), targets, self.smooth)
        return ce_loss + dice
