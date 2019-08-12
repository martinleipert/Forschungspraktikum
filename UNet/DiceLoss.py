"""

Original implementation taken from
https://github.com/usuyama/pytorch-unet

modified by 
Martin Leipert
martin.leipert@t-online.de

"""
import torch
import numpy as np


# Dice Loss
# Correlation between prediction and reference
def dice_loss(pred, target, smooth = 1., weights = None):
    pred = pred.contiguous()
    target = target.contiguous()

    nr_labels = pred.shape[1]

    loss = torch.zeros(1)

    if weights is not None:
        weights = np.float32(weights)

    for i in range(nr_labels):
        l_pred = pred[:, i, :, :]
        l_pred = l_pred.view(l_pred.size(0), -1)
        l_target = target[:, i, :, :]
        l_target = l_target.view(l_target.size(0), -1)

        inter = (l_pred * l_target).sum()

        cur_loss = 1 - ((2. * inter + smooth) / (l_pred.sum() + l_target.sum() + smooth))
        if weights is not None:
            cur_loss = cur_loss*weights[i]

        loss += cur_loss*(1 / nr_labels)
    
    return loss
