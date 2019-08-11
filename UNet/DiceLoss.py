"""

Original implementation taken from
https://github.com/usuyama/pytorch-unet

modified by 
Martin Leipert
martin.leipert@t-online.de

"""
import torch


# Dice Loss
# Correlation between prediction and reference
def dice_loss(pred, target, smooth = 1., weights = None):
    pred = pred.contiguous()
    target = target.contiguous()    

    nr_labels = pred.shape[1]

    sum_loss = torch.zeros(pred.shape[1])

    for i in range(nr_labels):
        l_pred = pred[:, i, :, :]
        l_pred = l_pred.view(l_pred.size(0), -1)
        l_target = target[:, i, :, :]
        l_target = l_target.view(l_target.size(0), -1)

        inter = (l_pred * l_target).sum()

        loss = 1 - ((2. * inter + smooth) / (l_pred.sum() + l_target.sum() + smooth))
        if weights:
            loss = loss*weights[i]

        sum_loss[i] = loss*(1 / nr_labels)
    
    return sum_loss.mean()
