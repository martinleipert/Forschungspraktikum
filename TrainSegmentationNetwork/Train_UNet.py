from collections import defaultdict
import torch.nn.functional as F
from UNet.DiceLoss import dice_loss
import time
import copy
import torch
# from UNet.pytorch_unet import UNet
from UNet.BatchNormUNet import UNet
# from UNet.ThirdUNet import UNet
from UNetLoader_dynamic import UNetDatasetDynamicMask
from torch.optim import lr_scheduler
from UNet.FocalLoss import FocalLoss2d

from matplotlib import pyplot
import numpy as np

"""
Martin Leipert
martin.leipert@fau.de

Stolen from 
https://github.com/usuyama/pytorch-unet
"""

SET_NAME = "mini_set"

FILE_LIST_TRAINING = "/home/martin/Forschungspraktikum/Testdaten/Segmentation_Sets/%s/training.txt" % SET_NAME
FILE_LIST_VALIDATION = "/home/martin/Forschungspraktikum/Testdaten/Segmentation_Sets/%s/validation.txt" % SET_NAME

BATCH_SIZE = 5
NUM_EPOCHS = 25
NUM_CLASS = 4

# FOCAL or BCE_DICE
UNET_LOSSFKT = "FOCAL"
LOAD_MODEL = False
SAVE_MODEL = False
MODEL_NAME = "unet_full_training.pth"



def main():
    torch.cuda.empty_cache()

    # Load with self written FIle loader
    training_data = UNetDatasetDynamicMask(FILE_LIST_TRAINING, region_select=True, augment=True)
    validation_data = UNetDatasetDynamicMask(FILE_LIST_VALIDATION, region_select=False)

    # Define the DataLoader
    trainloader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if LOAD_MODEL is True:
        model = torch.load(MODEL_NAME)
    else:
        model = UNet(NUM_CLASS)
    model.to(device)

    # freeze backbone layers
    # for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.5)

    """
    Here  it  could be separated into a method
    """

    start_time = time.time()

    optimizer = optimizer_ft

    for epoch in range(NUM_EPOCHS):

        # Start the training phase of an epoch
        epoch_start = time.time()

        print(f'\n{epoch_start - start_time} s elapsed\nEpoch {epoch + 1 }/{NUM_EPOCHS}')
        print('-' * 10)

        # exp_lr_scheduler.step()

        # Trainingphase
        model.train()
        epoch_samples = 0
        metrics = defaultdict(float)

        print("===== Training =====")
        # Imagewise Training
        with torch.set_grad_enabled(True):

            for images, masks in trainloader:
                images = images.to(device)
                masks = masks.to(device)

                # Set Gradients to zero
                optimizer.zero_grad()

                # forward
                # track history if only in train
                outputs = model.forward(images)

                # sum_set = np.sum(torch.sigmoid(outputs).cpu().detach().numpy(), axis=1)
                # numpy_set = torch.sigmoid(outputs).cpu().detach().numpy()
                # for i in range(sum_set.shape[0]):
                #     numpy_set[i, :, :, :] = np.divide(numpy_set[i, :, :, :], sum_set[i, :, :])
                # pyplot.imshow(numpy_set[0, 0, :, :], vmin=0, vmax=1)


                # TODO BCE seems to be broken ?
                # F.binary_cross_entropy_with_logits(outputs.float(), masks.float())
                loss = calc_loss(outputs, masks, metrics)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                # statistics
                epoch_samples += images.size(0)

        print_metrics(metrics, epoch_samples, 'train')

        # Validation phase
        model.eval()
        metrics = defaultdict(float)
        epoch_samples = 0
        print("===== Validation =====")
        with torch.set_grad_enabled(False):
            for images, masks in validationloader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                outputs = model(images)

                # TODO BCE seems to be broken ?
                loss = calc_loss(outputs, masks, metrics, all_losses = True)

                # statistics
                epoch_samples += images.size(0)

        print_metrics(metrics, epoch_samples, 'validation')

        """
        pred_array = np.array(outputs[0, 0, :, :].cpu().detach())
        mask_array = np.array(masks[0, 0, :, :].cpu().detach())

        figure = pyplot.figure(0)
        ax1 = figure.add_subplot(1, 2, 1)
        ax2 = figure.add_subplot(1, 2, 2)

        ax1.imshow(mask_array)
        ax2.imshow(pred_array)

        figure.show()
        # Evaluationphase
        model.eval()

        epoch_end = time.time()
        """

        if epoch % 5 == 0 and SAVE_MODEL:
            torch.save(model, MODEL_NAME)


    end_time = time.time()

    pass

# TODO functions and weightings in a dictionary

def calc_loss(pred, target, metrics, bce_weight=0.5, all_losses=False):

    # Fix -> TODO find out why necessary and remove
    target = target.double()
    pred = pred.double()
    pred = torch.sigmoid(pred)

    if (UNET_LOSSFKT == "FOCAL"):
        loss = FocalLoss2d().forward(pred, target)
        metrics['focal'] += loss.data.cpu().numpy() * target.size(0)

    elif (UNET_LOSSFKT == "BCE_DICE"):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = dice_loss(pred, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)
        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)

    if all_losses:
        floss = FocalLoss2d().forward(pred, target)
        metrics['focal'] += floss.data.cpu().numpy() * target.size(0)

        bce = F.binary_cross_entropy_with_logits(pred, target)
        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)

        dice = dice_loss(pred, target)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)

        loss = bce * 1./3 + dice * 1./3 + floss * 1./3

    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def plot_helper(dataset, selector):
    sum_set = np.sum(torch.sigmoid(dataset).cpu().detach().numpy(), axis=1)

    numpy_set = torch.sigmoid(dataset).cpu().detach().numpy()

    for i in range(sum_set.shape[0]):
        numpy_set[i, :, :, :] = np.divide(numpy_set[i, :, :, :], sum_set[i, :, :])
        pass

    pyplot.imshow(numpy_set[0, selector, :, :], vmin=0, vmax=1)


def mask_helper(image, mask):
    pyplot.imshow(np.where(mask[0, 3, :, :].cpu() == 1, image[0, 0, :, :].cpu(), 0))


if __name__ == '__main__':
    main()
