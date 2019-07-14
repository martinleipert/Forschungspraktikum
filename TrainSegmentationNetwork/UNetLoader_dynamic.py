from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TF
from PIL import Image, ImageDraw, ImageFile
import numpy as np
import os
from xml.etree import ElementTree as ET
# For random region selection
import random
import cv2
from torch import tensor
from Augmentations.Augmentations import weak_augmentation

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, ElasticTransform
)


from matplotlib import pyplot as plt

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
ImageFile.LOAD_TRUNCATED_IMAGES = True

REGION_TYPES = {
    "Background": 0,
    "TextRegion": 1,
    "ImageRegion": 2,
    "GraphicRegion": 3
}


# Helper Method for bounding box extraction
def extract_bounding_box(pt_arr):

    pt_arr = np.array(pt_arr)
    x_min, y_min = np.min(pt_arr, 0)
    x_max, y_max = np.max(pt_arr, 0)

    bb_coordinates = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    bb_parameters = (x_min, y_min, x_max - x_min, y_max - y_min)
    return bb_parameters


"""
Load the mask images
which contain the information where an object is located in binary format
"""


def dynamic_mask_loader(path, augmentation=None, region_select=False, p_region_select=0.9):

    dir_name = os.path.dirname(path)
    filename = os.path.basename(path).rsplit('.jpg')[0]
    xml_filename = f"{filename}.xml"
    xml_path = os.path.join(dir_name, 'page', xml_filename)

    try:
        xml_tree = ET.parse(xml_path)
    except Exception as e:
        pass
    root = xml_tree.getroot()
    ns = '{' + SCHEMA + '}'

    # Load mask
    image = Image.open(path)
    image.load()
    image = image.convert('RGB')

    # Contruct mask array
    mask_array = np.int8(np.zeros(list([image.size[1], image.size[0]]) + [len(REGION_TYPES)]))

    all_els = []

    # Generate masks for the regions
    for key, value in REGION_TYPES.items():
        els = root.findall("*/" + ns + key)

        for el in els:
            points = el.find(ns + 'Coords').get('points')
            fix_pts = tuple(map(lambda x: tuple(map(int, x.split(','))), points.split(' ')))

            img = Image.new('L', [image.size[0], image.size[1]], 0)
            ImageDraw.Draw(img).polygon(fix_pts, fill=1, outline=1, )
            mask = np.array(img)

            mask_array[:, :, value] = np.logical_or(mask, mask_array[:, :, value])

        # Add to all pixels where there is a value so we can mask out afterwards
        mask_array[:, :, 0] = mask_array[:, :, 0] + mask_array[:, :, value]
        all_els.extend(els)

    # Mask out the background
    mask_array[:, :, 0] = np.where(mask_array[:, :, 0] == 0, 1, 0)
    mask_array = np.uint8(mask_array)

    # If a random region get's masked out
    if region_select:
        randint = random.randint(1, 100)

        # TODO: Use different snippet sizes so validation loss gets reduced
        if randint < 75:
            elidx = random.randint(0, len(all_els)-1)
            element = all_els[elidx]
            points = element.find(ns + 'Coords').get('points')
            fix_pts = tuple(map(lambda x: tuple(map(int, x.split(','))), points.split(' ')))

            # Get the regions bounding box
            xbb, ybb, wbb, hbb = extract_bounding_box(fix_pts)

            if hbb > 25 and wbb > 25:

                factor = 0.25

                sub_h = hbb*factor
                sub_w = wbb*factor

                xbb = xbb-sub_w
                ybb = ybb-sub_h
                wbb = wbb + 2*sub_w
                hbb = hbb + 2*sub_h

                image = image.crop((xbb, ybb, xbb+wbb, ybb+hbb))

                # Apply the crop to the mask parts
                cropped_mask = np.zeros((image.size[1], image.size[0], mask_array.shape[2]))
                for i in range(mask_array.shape[2]):
                    tmp_im = Image.fromarray(mask_array[:, :, i])
                    tmp_im = tmp_im.crop((xbb, ybb, xbb + wbb, ybb + hbb))

                    cropped_mask[:, :, i] = np.array(tmp_im)

                horst = 1
                mask_array = cropped_mask
                pass

    # Augmentation-independent Transformation -> TO Tensor
    trans3 = TF.ToTensor()


    if augmentation is None:
        trans1 = TF.Resize(256)
        trans2 = TF.CenterCrop(224)

        trafo_img = trans3(trans2(trans1(image)))
        trafo_mask = np.array([np.array(trans3(trans2(trans1(Image.fromarray(mask_array[:, :, i]))))) for i in range(np.shape(mask_array)[2])])
    else:

        transformed = augmentation(image=np.array(image), mask=mask_array)

        trafo_img = trans3(transformed['image'])
        trafo_mask = trans3(transformed['mask'])

        pass

    # print("%s, %s, %s, %s" % (path, str(image.size), image.mode, str(trafo_mask.shape)))

    return trafo_img, trafo_mask


"""
A Dataset containing data from a list which is built of tuples:
"""


class UNetDatasetDynamicMask(Dataset):
    def __init__(self, file_path, transform=None, data_set_loader=dynamic_mask_loader, augment=True,
                 region_select=False, augmentation=weak_augmentation):
        self.input_images = []
        self.data_set_loader = data_set_loader
        self.augment = augment
        self.augmentation = augmentation()
        self.region_select = region_select

        with open(file_path) as open_file:
            lines = open_file.read()
            splitted_lines = lines.split('\n')

            for line in splitted_lines:
                if line == "":
                    continue
                img = line.strip()

                self.input_images.append(img)

        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image_path = self.input_images[idx]

        image, mask = self.data_set_loader(image_path, region_select=self.region_select)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        if self.augment is True:
            image, mask = self.augmentation(image, mask)

            image = tensor(np.float32(image))
            mask = tensor(mask)

        # Todo: Remove!
        trafo = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        image = trafo(image)

        return [image, mask, image_path]


if __name__ == "__main__":
    from Augmentations.Augmentations import moderate_augmentation
    from matplotlib import pyplot

    file = "/home/martin/Forschungspraktikum/Testdaten/Transkribierte_Notarsurkunden/" \
           "notarskurkunden_mom_restored/0001_ABP_14341123_PfA-Winzer-U-0002-0_r.jpg"
    trafo_img, trafo_mask = dynamic_mask_loader(file, region_select=False, augmentation=moderate_augmentation())
    pyplot.imshow(trafo_img.numpy()[0, :, :])
    pyplot.imshow(trafo_mask.numpy()[0, :, :])
    pass
