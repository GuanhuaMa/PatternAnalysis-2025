import torch
import numpy as np

def calculate_dice_score(pred_binary, true_mask):
    """calculate Dice similarity coefficient"""

    if isinstance(pred_binary, torch.Tensor):
        pred_binary = pred_binary.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()

    pred_binary = pred_binary.flatten()
    true_mask = true_mask.flatten()

    # + 1e-6 prevent the denominator is 0
    intersection = (pred_binary * true_mask).sum()
    dice_score = (2. * intersection + 1e-6) / (pred_binary.sum() + true_mask.sum() + 1e-6)

    return dice_score