from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image, ImageDraw, ImageFile
import numpy as np
import os
from xml.etree import ElementTree as ET
# For random region selection
import random

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

    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()
    ns = '{' + SCHEMA + '}'

    image = Image.open(path)

    image.load()
    image = image.convert('RGB')

    img_array = np.int8(np.zeros(list(image.size) + [len(REGION_TYPES)]))

    all_els = []

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
        all_els.extend(els)
    # Mask out the background
    img_array[:, :, 0] = np.where(img_array[:, :, 0] == 0, 1, 0)


    # Augmentation-independent Transformation -> TO Tensor
    trans3 = transforms.ToTensor()

    if region_select:
        element = all_els[random.randint(0, len(all_els)-1)]
        points = element.find(ns + 'Coords').get('points')
        fix_pts = tuple(map(lambda x: tuple(map(int, x.split(','))), points.split(' ')))

        xbb, ybb, wbb, hbb = extract_bounding_box(fix_pts)

        factor = 0.15

        sub_h = hbb*factor
        sub_w = wbb*factor

        xbb = xbb-sub_w
        ybb = ybb-sub_h
        wbb = wbb + 2*sub_w
        hbb = hbb + 2*sub_h

        image = image.crop((xbb, ybb, xbb+wbb, ybb+hbb))

        mask_array = np.zeros((image.size[1], image.size[0], img_array.shape[2]))
        for i in range(img_array.shape[2]):
            tmp_im = Image.fromarray(img_array[:, :, i])
            tmp_im = tmp_im.crop((xbb, ybb, xbb + wbb, ybb + hbb))

            mask_array[:, :, i] = np.array(tmp_im)
        # TODO CROP such that the regions are inside

        pass

    # todo PROCEEED HERE

    if augmentation is None:
        trans1 = transforms.Resize(256)
        trans2 = transforms.CenterCrop(224)

        trafo_img = trans3(trans2(trans1(image)))

    else:
        trafo_img = augmentation(trafo_img)
        pass

    mask_list = list()
    for i in range(img_array.shape[2]):
        tmp_im_array = Image.fromarray(img_array[:, :, i])
        if augmentation is None:
            mask_list.append(trans2(trans1(tmp_im_array)))
        else:
            mask_list.append(tmp_im_array)
            pass
    mask_list = [np.asarray(im) for im in mask_list]

    trafo_mask = np.zeros(list(mask_list[0].shape) + [img_array.shape[2]])

    for i in range(trafo_mask.shape[2]):
        trafo_mask[:, :, i] = mask_list[i]

    trafo_mask = trans3(trafo_mask)

    # print("%s, %s, %s, %s" % (path, str(image.size), image.mode, str(trafo_mask.shape)))

    return trafo_img, trafo_mask


"""
A Dataset containing data from a list which is built of tuples:
"""


class UNetDatasetDynamicMask(Dataset):
    def __init__(self, file_path, transform=None, data_set_loader=dynamic_mask_loader, augment=False,
                 region_select=False):
        self.input_images = []
        self.target_masks = []
        self.data_set_loader = data_set_loader
        self.augment = augment
        self.region_select = region_select

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

        image, mask = self.data_set_loader(image, region_select=self.region_select)

        """
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        """

        return [image, mask]


if __name__ == "__main__":
    file = "/home/martin/Forschungspraktikum/Testdaten/Transkribierte_Notarsurkunden/" \
           "notarskurkunden_mom_restored/0001_ABP_14341123_PfA-Winzer-U-0002-0_r.jpg"
    trafo_img, trafo_mask = dynamic_mask_loader(file, region_select=True)
    pass
