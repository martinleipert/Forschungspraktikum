from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image, ImageDraw
import numpy as np
import os


"""
Martin Leipert
martin.leipert@fau.de

Stolen from 
https://github.com/usuyama/pytorch-unet
"""

"""
Loader for the RGB Images from the Directory

"""

SCHEMA = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

Image.MAX_IMAGE_PIXELS = 933120000

REGION_TYPES = {
    "Background": 0,
    "TextRegion": 1,
    "ImageRegion": 2,
    "GraphicRegion": 3
}


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


def dynamic_mask_loader(path):

    dir_name = os.path.dirname(path)
    filename = os.path.basename(path).rsplit('.jpg')
    xml_filename = f"{filename}.xml"
    xml_path = os.path.join(dir_name, 'page', xml_filename)

    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()
    ns = f'{SCHEMA}'

    image = Image.open(path)

    for i in range(3):
        try:
            image.load()
        except Exception as e:
            pass

    img_array = np.float32(np.zeros(list(image.size) + [len(REGION_TYPES)]))

    for key, value in REGION_TYPES.items():
        els = root.findall("*/" + ns + key)

        for el in els:
            points = el.find(ns + 'Coords').get('points')
            fix_pts = tuple(map(lambda x: tuple(map(int, x.split(','))), points.split(' ')))

            img = Image.new('L', [image.size[1], image.size[0]], 0)
            ImageDraw.Draw(img).polygon(fix_pts, fill=1, outline=1, )
            mask = np.array(img)

            img_array[:, :, value] = np.logical_or(mask, img_array[:, :, value])

        # Add to all pixels where there is a value so we can mask out afterwards
        img_array[:, :, 0] = img_array[:, :, 0] + img_array[:, :, value]

    # Mask out the background
    img_array[:, :, 0] = np.where(img_array[:, :, 0] == 0, 1, 0)

    trans1 = transforms.Resize(256)
    trans2 = transforms.CenterCrop(224)
    trans3 = transforms.ToTensor()

    trafo_img = trans3(trans2(trans1(image)))
    trafo_mask = trans3(trans2(trans1(img_array)))

    return trafo_img, trafo_mask


"""
A Dataset containing data from a list which is built of tuples:
"""


class UNetDatasetDynamicMask(Dataset):
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
