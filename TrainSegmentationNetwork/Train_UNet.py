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
import logging

import sys
from matplotlib import pyplot
import numpy as np

# TODO implement logging -> Loss and recognition
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
NUM_CLASS = 4
NUM_EPOCHS = 45
# Step size of Learning rate decay
LR_STEP_SIZE = 15

# FOCAL or BCE_DICE
UNET_LOSSFKT = "FOCAL"
LOAD_MODEL = False
SAVE_MODEL = False
MODEL_NAME = "unet_full_training.pth"


TRAIN_LOSSES_WEIGHTING = {
    "BCE_LOSS" : 0,
    "DICE_LOSS" : 0,
    "FOCAL_LOSS" : 1
}

VAL_LOSSES_WEIGHTING = {
    "BCE_LOSS" : 0.33,
    "DICE_LOSS" : 0.33,
    "FOCAL_LOSS" : 0.33
}

__LOG_PATH = f"Training_Log_{MODEL_NAME}_{SET_NAME}.txt"
formatter = logging.Formatter('%(asctime)s\n%(message)s')

file_handler = logging.FileHandler(__LOG_PATH)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
cmd_handler = logging.StreamHandler(sys.stdout)
cmd_handler.setLevel(logging.DEBUG)
cmd_handler.setFormatter(formatter)

__LOGGER__ = logging.Logger("Training Logger")
__LOGGER__.addHandler(file_handler)
__LOGGER__.addHandler(cmd_handler)


def main():
    torch.cuda.empty_cache()

    # Load with self written FIle loader
    training_data = UNetDatasetDynamicMask(FILE_LIST_TRAINING, region_select=True, augment=True)
    validation_data = UNetDatasetDynamicMask(FILE_LIST_VALIDATION, region_select=False)

    # Define the DataLoader
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if LOAD_MODEL is True:
        model = UNet(NUM_CLASS)
        model.load_state_dict(torch.load("TrainedModels/%s" % MODEL_NAME))
    else:
        model = UNet(NUM_CLASS)
    model.to(device)

    # freeze backbone layers
    # for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.5)

    """
    Here  it  could be separated into a method
    """
    train_losses = []
    validation_losses = []

    """
    Plotting
    """
    # Prepare the plots
    loss_fig = pyplot.figure(1, figsize=(10, 7))
    loss_ax = loss_fig.add_subplot(111)
    loss_ax.set_xlabel("Epochs")
    loss_ax.set_ylabel("")
    loss_ax.set_ylim([0, 1])
    loss_ax.set_title("Loss-Curves of %s" % MODEL_NAME)

    train_loss_curve, = loss_ax.plot([], 'b-', label="Training Loss")
    validation_loss_curve, = loss_ax.plot([], 'r-', label="Validation Loss")
    loss_fig.show()
    pyplot.pause(0.05)

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):

        # Start the training phase of an epoch
        epoch_start = time.time()

        print(f'\n{epoch_start - start_time} s elapsed\nEpoch {epoch + 1 }/{NUM_EPOCHS}')
        print('-' * 10)

        exp_lr_scheduler.step()

        # Trainingphase
        model.train()
        epoch_samples = 0
        metrics = defaultdict(float)

        __LOGGER__.info("===== Training =====")
        # Imagewise Training
        with torch.set_grad_enabled(True):

            for images, masks, image_paths in train_loader:

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
                loss = calc_loss(outputs, masks, metrics, TRAIN_LOSSES_WEIGHTING)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                # statistics
                epoch_samples += images.size(0)

        print_metrics(metrics, epoch_samples, 'train')

        train_loss = metrics['loss']
        train_losses.append(train_loss)
        train_loss_curve.set_xdata(range(len(train_losses)))
        train_loss_curve.set_ydata(np.array(train_losses))

        # Validation phase
        model.eval()
        metrics = defaultdict(float)
        epoch_samples = 0
        __LOGGER__.info("===== Validation =====")
        with torch.set_grad_enabled(False):
            for images, masks, image_paths in validation_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                outputs = model(images)

                loss = calc_loss(outputs, masks, metrics, VAL_LOSSES_WEIGHTING)

                # statistics
                epoch_samples += images.size(0)

        print_metrics(metrics, epoch_samples, 'validation')

        val_loss = metrics['loss']
        validation_losses.append(val_loss)
        validation_loss_curve.set_xdata(range(len(validation_losses)))
        validation_loss_curve.set_ydata(np.array(validation_losses))
        loss_ax.set_xlim((0, len(validation_losses) - 1))
        pyplot.pause(0.05)
        loss_fig.savefig("%s_%s.png" % (MODEL_NAME, SET_NAME), dpi=200)

        if epoch % 5 == 4 and SAVE_MODEL:
            torch.save_state_dict(model.state_dict(), "TrainedModels/%s" % MODEL_NAME)

    end_time = time.time()
    torch.save_state_dict(model.state_dict(), "TrainedModels/%s" % MODEL_NAME)

    pass


# TODO functions and weightings in a dictionary
def calc_loss(pred, target, metrics, losses_weighting):

    # Fix -> TODO find out why necessary and remove
    target = target.double()
    pred = torch.sigmoid(pred.double())

    focal_weight = losses_weighting["FOCAL_LOSS"]
    dice_weight = losses_weighting["DICE_LOSS"]
    bce_weight = losses_weighting["BCE_LOSS"]

    total_weight = focal_weight + dice_weight + bce_weight

    focal_loss, bce_loss, dice = (0, 0, 0)

    if focal_weight > 0:
        focal_loss = FocalLoss2d().forward(pred, target)
        metrics['focal'] += focal_loss.data.cpu().numpy() * target.size(0)

    if dice_weight > 0:
        dice = dice_loss(pred, target)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)

    if bce_weight > 0:
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        metrics['bce'] += bce_loss.data.cpu().numpy() * target.size(0)

    sum_loss = (bce_loss * bce_weight + dice * dice_weight + focal_loss * focal_weight) / total_weight

    metrics['loss'] += sum_loss.data.cpu().numpy() * target.size(0)

    return sum_loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    __LOGGER__.info("{}: {}".format(phase, ", ".join(outputs)))


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
