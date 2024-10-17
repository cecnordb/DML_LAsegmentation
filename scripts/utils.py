import torch.nn as nn

def dice_coefficient(pred, target, smooth=1.):
    pred = pred > 0.5  # Thresholding to get binary predictions
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def dice_loss(pred, target, smooth = 1.):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        ce_loss = self.ce_loss(outputs, targets)
        dice = dice_loss(outputs, targets)
        return ce_loss + dice
