from collections import defaultdict
import torch.nn.functional as F
from UNet.loss import dice_loss
import time
import copy
import torch
# from UNet.pytorch_unet import UNet
from UNet.BetterUNet import UNet
from UNetLoader_dynamic import UNetDatasetDynamicMask
from torchvision import transforms

from matplotlib import pyplot
import numpy as np

"""
Martin Leipert
martin.leipert@fau.de

Stolen from 
https://github.com/usuyama/pytorch-unet
"""


FILE_LIST_TRAINING = "/home/martin/Forschungspraktikum/Testdaten//Transkribierte_Notarsurkunden/siegel_files.txt"
BATCH_SIZE = 5


def main():
    torch.cuda.empty_cache()
    """
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
    ])
    """

    # Load with self written FIle loader
    training_data = UNetDatasetDynamicMask(FILE_LIST_TRAINING, region_select=True)     #, transform=trans)
    # validation_data = ImageFilelist('.', VALIDATION_SET)
    # test_data = ImageFilelist('.', TEST_SET)

    # Define the DataLoader
    trainloader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
    # validationloader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)
    # testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_class = 4
    model = UNet(num_class, depth=5)
    model.to(device)

    # freeze backbone layers
    # for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    # model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=60)


    """
    Here  it  could be separated into a method
    """
    num_epochs = 25

    start_time = time.time()

    optimizer = optimizer_ft

    for epoch in range(num_epochs):

        # Start the training phase of an epoch
        epoch_start = time.time()

        print(f'\n{epoch_start - start_time} s elapsed\nEpoch {epoch + 1 }/{num_epochs}')
        print('-' * 10)

        # Trainingphase
        model.train()
        epoch_samples = 0
        metrics = defaultdict(float)


        for images, masks in trainloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(images)
                loss = calc_loss(outputs, masks, metrics)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            epoch_samples += images.size(0)


        print_metrics(metrics, epoch_samples, 'train')
        epoch_loss = metrics['loss'] / epoch_samples

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


    end_time = time.time()

    pass


def calc_loss(pred, target, metrics, bce_weight=0.5):
    # Fix -> TODO find out why necessary and remove
    target = target.double()
    pred = pred.double()

    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    main()
