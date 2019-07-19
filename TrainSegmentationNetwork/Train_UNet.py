from collections import defaultdict
import torch.nn.functional as F
from UNet.DiceLoss import dice_loss
import time
import torch
from UNet.BatchNormUNet import UNet
from TrainSegmentationNetwork.UNetLoader_dynamic import UNetDatasetDynamicMask
from torch.optim import lr_scheduler
from UNet.FocalLoss import FocalLoss2d
import logging
from argparse import ArgumentParser
from Augmentations.Augmentations import weak_augmentation, moderate_augmentation, heavy_augmentation

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


DIR_ROOT = "/home/martin/Forschungspraktikum/Testdaten/Segmentation_Sets"

BATCH_SIZE = 5
NUM_CLASS = 4

# Precomputed weights for the classes according to their occurence frequency in the set
WEIGHTS = [
    0.4124711540461253,
    0.8165794359287605,
    6.900447952756638,
    4.853207270061199
]

VAL_LOSSES_WEIGHTING = {
    "BCE_LOSS": 0.33,
    "DICE_LOSS": 0.33,
    "FOCAL_LOSS": 0.33
}

formatter = logging.Formatter('%(asctime)s\n%(message)s')

cmd_handler = logging.StreamHandler(sys.stdout)
cmd_handler.setLevel(logging.DEBUG)
cmd_handler.setFormatter(formatter)

__LOGGER__ = logging.Logger("Training Logger")
__LOGGER__.addHandler(cmd_handler)


def main():
    arg_parser = ArgumentParser("Train a UNet with the parameters given by the Parser")
    arg_parser.add_argument("SET_NAME", help="Name of the set used for training")
    arg_parser.add_argument("BCE", type=float, help="Weight of the bce loss function")
    arg_parser.add_argument("DICE", type=float, help="Weight of the dice loss function")
    arg_parser.add_argument("FOCAL", type=float, help="Weight of the focal loss function")
    arg_parser.add_argument("AUGMENTATION", type=str, help="Selected Augmentation: 'WEAK', 'MODERATE', 'HEAVY'")
    arg_parser.add_argument("--learningRate", type=float, default=1e-3)
    arg_parser.add_argument("--epochs", type=int, default=30)
    arg_parser.add_argument("--lrStep", type=int, default=10)
    arg_parser.add_argument("--regionSelect", type=bool, nargs='?', default=False
                            default="Use the region select to counter class imbalance")

    parsed_args = arg_parser.parse_args()

    augmentation_fct = parsed_args.AUGMENTATION

    train_fresh = False

    set_name = parsed_args.SET_NAME
    model_name = "unet_%s_training.pth" % set_name

    learning_rate = parsed_args.learningRate
    num_epochs = parsed_args.epochs
    # Step size of Learning rate decay
    lr_step_size = parsed_args.lrStep
    region_select = parsed_args.regionSelect

    train_losses_weighting = {
        "BCE_LOSS": parsed_args.BCE,
        "DICE_LOSS": parsed_args.DICE,
        "FOCAL_LOSS": parsed_args.FOCAL
    }

    file_list_training = f"%s/%s/training.txt" % (DIR_ROOT, set_name)
    file_list_validation = f"%s/%s/validation.txt" % (DIR_ROOT, set_name)

    __LOG_PATH = f"TrainingLogs/Training_Log_{model_name}_{set_name}.txt"
    file_handler = logging.FileHandler(__LOG_PATH)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    __LOGGER__.addHandler(file_handler)

    __LOGGER__.info(f"Start Training of {model_name}\n"
                    f"Training on {set_name}\n"
                    f"Learning Rate: {learning_rate}\n"
                    f"Batch size: {BATCH_SIZE}\n"
                    f"Epochs: {num_epochs}\n"
                    f"Learning Rate Step Size: {lr_step_size}\n"
                    f"Loss composition: {train_losses_weighting}\n"
                    f"Augmentation function: {augmentation_fct}\n"
                    f"Region select: {region_select}\n")

    """
    Prepare the data
    """
    if augmentation_fct == "WEAK":
        augmentation_fct = weak_augmentation
    elif augmentation_fct == "MODERATE":
        augmentation_fct = moderate_augmentation
    elif augmentation_fct == "HEAVY":
        augmentation_fct = heavy_augmentation

    torch.cuda.empty_cache()

    # Load with self written FIle loader
    training_data = UNetDatasetDynamicMask(file_list_training, region_select=region_select,
                                           augmentation=augmentation_fct)
    validation_data = UNetDatasetDynamicMask(file_list_validation, region_select=False)

    # Define the DataLoader
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if train_fresh is True:
        model = UNet(NUM_CLASS)
        model.load_state_dict(torch.load("TrainedModels/%s" % model_name))
    else:
        model = UNet(NUM_CLASS)
    model.to(device)

    # freeze backbone layers
    # for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.5)

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
    loss_ax.set_title("Loss-Curves of %s" % model_name)

    train_loss_curve, = loss_ax.plot([], 'b-', label="Training Loss")
    validation_loss_curve, = loss_ax.plot([], 'r-', label="Validation Loss")
    loss_fig.show()
    pyplot.pause(0.05)

    start_time = time.time()

    for epoch in range(num_epochs):

        # Start the training phase of an epoch
        epoch_start = time.time()

        __LOGGER__.info(f'\n{epoch_start - start_time} s elapsed\nEpoch {epoch + 1 }/{num_epochs}')
        __LOGGER__.info('-' * 10)

        exp_lr_scheduler.step()

        # Training phase
        model.train()
        epoch_samples = 0
        metrics = defaultdict(float)

        __LOGGER__.info("===== Training =====")
        # Image-wise Training
        for images, masks, image_paths in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            # Set Gradients to zero
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs = model.forward(images)

            loss = calc_loss(outputs, masks, metrics, train_losses_weighting)

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
        with torch.no_grad():
            for images, masks, image_paths in validation_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                outputs = model(images)

                loss = calc_loss(outputs, masks, metrics, VAL_LOSSES_WEIGHTING).detach()

                # statistics
                epoch_samples += images.size(0)

        print_metrics(metrics, epoch_samples, 'validation')

        val_loss = metrics['loss']
        validation_losses.append(val_loss)
        validation_loss_curve.set_xdata(range(len(validation_losses)))
        validation_loss_curve.set_ydata(np.array(validation_losses))
        loss_ax.set_xlim((-1, len(validation_losses)))
        pyplot.pause(0.05)
        loss_fig.savefig("TrainingLogs/%s_%s.png" % (model_name, set_name), dpi=200)

        if epoch % 5 == 4:
            torch.save(model.state_dict(), "TrainedModels/%s" % model_name)

    torch.save(model.state_dict(), "TrainedModels/%s" % model_name)

    pass


def calc_loss(pred, target, metrics, losses_weighting):

    target = target.double()
    pred = torch.sigmoid(pred.double())

    focal_weight = losses_weighting["FOCAL_LOSS"]
    dice_weight = losses_weighting["DICE_LOSS"]
    bce_weight = losses_weighting["BCE_LOSS"]

    total_weight = focal_weight + dice_weight + bce_weight

    focal_loss, bce_loss, dice = (0, 0, 0)

    if focal_weight > 0:
        focal_loss = FocalLoss2d(gamma=2, weight=WEIGHTS).forward(pred, target)
        metrics['focal'] += focal_loss.data.cpu().detach().numpy() * target.size(0)

    if dice_weight > 0:
        dice = dice_loss(pred, target)
        metrics['dice'] += dice.data.cpu().detach().numpy() * target.size(0)

    if bce_weight > 0:
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        metrics['bce'] += bce_loss.data.cpu().detach().numpy() * target.size(0)

    sum_loss = (bce_loss * bce_weight + dice * dice_weight + focal_loss * focal_weight) / total_weight

    metrics['loss'] += sum_loss.data.cpu().detach().numpy() * target.size(0)

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
