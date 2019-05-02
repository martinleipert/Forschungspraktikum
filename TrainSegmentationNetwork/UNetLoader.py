from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np

"""
Martin Leipert
martin.leipert@fau.de

Stolen from 
https://github.com/usuyama/pytorch-unet
"""

"""
Loader for the RGB Images from the Directory

"""


def default_loader(path):

    image = Image.open(path)

    for i in range(3):
        try:
            image.load()
        except Exception as e:
            pass

    image = image.convert('RGB')
    trans1 = transforms.Resize(256)
    trans2 = transforms.CenterCrop(224)
    trans3 = transforms.ToTensor()

    transformed = trans3(trans2(trans1(image)))

    image.close()

    return transformed

"""
Load the mask images
which contain the information where an object is located in binary format
"""


def mask_loader(path):
    image = Image.open(path)

    for i in range(3):
        try:
            image.load()
        except Exception as e:
            pass

    trans1 = transforms.Resize(256)
    trans2 = transforms.CenterCrop(224)
    image = trans2(trans1(image))
    arr = np.array(image)

    new_image = np.zeros([np.shape(arr)[0], np.shape(arr)[1], 2])
    new_image[:, :, 0] = np.where(arr == 0, 1, 0)
    new_image[:, :, 1] = np.where(arr == 255, 1, 0)

    image = new_image

    trans3 = transforms.ToTensor()

    transformed = trans3(image)
    transformed.float()

    return transformed


"""
A Dataset containing data from a list which is built of tuples:
"""


class UNetDataset(Dataset):
    def __init__(self, file_path, transform=None, data_set_loader=default_loader, mask_loader=mask_loader):
        self.input_images = []
        self.target_masks = []
        self.data_set_loader = data_set_loader
        self.mask_loader = mask_loader

        with open(file_path) as open_file:
            lines = open_file.read()
            splitted_lines = lines.split('\n')

            for line in splitted_lines:
                if line is '':
                    continue
                img, mask = line.split(',')
                img = img.strip()
                mask = mask.strip()

                self.input_images.append(img)
                self.target_masks.append(mask)

        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]

        image = self.data_set_loader(image)
        mask = self.mask_loader(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return [image, mask]
