"""Module containing loss functions."""
from torch import nn


def average_loss(outputs, targets, pad_index):
    """
    Averge loss over a minin batch
    """
    batch_size = outputs.size(0)
    avg_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    for i in range(batch_size):
        y_pred, y_true = outputs[i], targets[i]
        avg_loss += criterion(y_pred, y_true)
    return avg_loss / batch_size
