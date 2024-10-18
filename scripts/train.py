from collections import defaultdict
from time import time
from scripts.utils import iou, dice_coefficient
import torch
import torch.nn.functional as F


def train_epoch(model, optimizer, loss_fn, train_loader, scheduler, device):
    model.train()
    train_loss = AccumulatingMetric()
    train_iou = AccumulatingMetric()
    train_dice = AccumulatingMetric()
    for batch in train_loader:
        input, target = batch
        input, target = input.to(device), target.to(device).float()
        optimizer.zero_grad()
        pred_logits = model(input)
        pred_logits = pred_logits.squeeze(1)
        loss = loss_fn(pred_logits, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        pred = torch.sigmoid(pred_logits)
        train_loss.add(loss.item())
        train_iou.add(iou(pred, target).cpu().item())
        train_dice.add(dice_coefficient(pred, target).cpu().item())

    return train_loss.avg(), train_iou.avg(), train_dice.avg()

def validate(model, loss_fn, val_loader, device, patch_size):
    model.eval()
    validation_loss = AccumulatingMetric()
    validation_iou = AccumulatingMetric()
    validation_dice = AccumulatingMetric()
    
    for batch in val_loader:
        # Here I need to separate the input into patches and then merge them after they have all been analyzed. Should I do this with overlap? Yes I should.
        input, target = batch
        input, target = input.to(device), target.to(device).float()
        with torch.no_grad():
            pred_logits = patched_forward(model, input, patch_size, device)
        pred_logits = pred_logits.squeeze(1)
        pred = torch.sigmoid(pred_logits)
        # Calculate the loss, iou and dice coefficient
        loss = loss_fn(pred_logits, target)

        validation_loss.add(loss.item())
        validation_iou.add(iou(pred, target).cpu().item())
        validation_dice.add(dice_coefficient(pred, target).cpu().item())
        
    return validation_loss.avg(), validation_iou.avg(), validation_dice.avg() 

def patched_forward(model, input, patch_size, device, overlap=0.5):
    # This function will take an input image and split it into patches, then run the model on each patch and merge the results.
    batch_size, channels, z, y, x = input.shape
    pz, py, px = patch_size
    stride_z, stride_y, stride_x = int(pz * (1 - overlap)), int(py * (1 - overlap)), int(px * (1 - overlap))

    # Calculate needed padding
    pad_z = (pz - z % pz) if z % pz != 0 else 0
    pad_y = (py - y % py) if y % py != 0 else 0
    pad_x = (px - x % px) if x % px != 0 else 0
    input_padded = F.pad(input, (0, pad_x, 0, pad_y, 0, pad_z))
    padded_z, padded_y, padded_x = input_padded.shape[2:]

    output = torch.zeros((batch_size, channels, padded_z, padded_y, padded_x)).to(device)
    weight_map = torch.zeros((batch_size, channels, padded_z, padded_y, padded_x)).to(device)

    for z_start in range(0, padded_z - pz + 1, stride_z):
        for y_start in range(0, padded_y - py + 1, stride_y):
            for x_start in range(0, padded_x - px + 1, stride_x):
                z_end, y_end, x_end = z_start + pz, y_start + py, x_start + px
                
                patch = input_padded[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
                patch_output = model(patch)
                # Add the output of the patch to the combined output
                output[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += patch_output

                # Keep track of how many values has been added to each region so that we can average them later
                weight_map[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += 1

    # Avoid division by zero in case some areas are not covered by patches
    weight_map = torch.where(weight_map == 0, torch.ones_like(weight_map), weight_map)

    output /= weight_map

    # Remove padding
    output = output[:, :, :z, :y, :x]
    return output


def train(model, optimizer, loss_fn, train_loader, val_loader, scheduler, device, epochs, patch_size, validation_freq=1):
    print(f"Starting training on device {device}...")
    model.to(device)
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)

    for epoch in range(epochs):
        start_time = time()
        train_loss, train_iou, train_dice = train_epoch(model, optimizer, loss_fn, train_loader, scheduler, device)
        train_metrics["loss"].append(train_loss)
        train_metrics["iou"].append(train_iou)
        train_metrics["dice"].append(train_dice)
        print(f"Epoch {epoch + 1} of {epochs} took {time() - start_time:.2f}s | Training: loss={train_loss:.4f}, iou={train_iou:.4f}, dice={train_dice:.4f}")
    
        if epoch % validation_freq == validation_freq - 1 or epoch == epochs - 1:
            val_loss, val_iou, val_dice = validate(model, loss_fn, val_loader, device, patch_size)
            val_metrics["loss"].append(val_loss)
            val_metrics["iou"].append(val_iou)
            val_metrics["dice"].append(val_dice)
            print(f"Validation: loss={val_loss:.4f}, iou={val_iou:.4f}, dice={val_dice:.4f}")

    return train_metrics, val_metrics


class AccumulatingMetric:
    """Accumulate samples of a metric and automatically keep track of the number of samples."""

    def __init__(self):
        self.metric = 0.0
        self.counter = 0

    def add(self, value):
        self.metric += value
        self.counter += 1

    def avg(self):
        return self.metric / self.counter