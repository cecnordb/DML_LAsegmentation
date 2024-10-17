from collections import defaultdict
from time import time
from scripts.utils import iou, dice_coefficient

def train_epoch(model, optimizer, loss_fn, train_loader, scheduler, device):
    model.train()
    train_loss = AccumulatingMetric()
    train_iou = AccumulatingMetric()
    train_dice = AccumulatingMetric()
    for batch in train_loader:
        input, target = batch
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(input)
        pred = pred.squeeze(1)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        train_loss.add(loss.item())
        train_iou.add(iou(pred, target))
        train_dice.add(dice_coefficient(pred, target))

    return train_loss.avg(), train_iou.avg(), train_dice.avg()

def validate(model, loss_fn, val_loader, device):
    model.eval()
    validation_loss = AccumulatingMetric()
    validation_iou = AccumulatingMetric()
    validation_dice = AccumulatingMetric()
    
    for batch in val_loader:
        input, target = batch
        input, target = input.to(device), target.to(device)
        pred = model(input)
        pred = pred.squeeze(1)
        loss = loss_fn(pred, target)
        validation_loss.add(loss.item())
        validation_iou.add(iou(pred, target))
        validation_dice.add(dice_coefficient(pred, target))
        
    return validation_loss.avg(), validation_iou.avg(), validation_dice.avg() 

def train(model, optimizer, loss_fn, train_loader, val_loader, scheduler, device, epochs, validation_freq=1):
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
            val_loss, val_iou, val_dice = validate(model, loss_fn, val_loader, device)
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