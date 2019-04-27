"""
Source: https://github.com/pytorch/vision/issues/81
"""

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import os.path
import re
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import random

Image.LOAD_TRUNCATED_IMAGES = True


def default_loader(path):

	image = Image.open(path)

	for i in range(3):
		try:
			image.load()
		except Exception as e:
			pass

	return image.convert('RGB')


def scale_loader(path, augmentation=None):

	image = default_loader(path)

	if augmentation:
		try:
			image = augmentation(image=image)["image"]
		except Exception as e:
			pass

	# image.thumbnail((224, 224), Image.ANTIALIAS)
	trans1 = transforms.Resize(256)
	trans2 = transforms.CenterCrop(224)
	trans3 = transforms.ToTensor()
	image = trans3(trans2(trans1(image)))
	return image


def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n ...(same to caffe's filelist)
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			try:
				stripped_line = line.strip()
				# impath, imlabel = stripped_line.split()
				impath, imlabel = re.search("(.*?)\s+(\d)", stripped_line).groups()
				imlist.append((impath, int(imlabel)))
			except Exception as e:
				pass
			if os.path.isdir(impath):
				pass

	return imlist


def unlabeled_flist_reader(flist):
	"""
	flist format: impath\nimpath\n ...
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			try:
				stripped_line = line.strip()
				impath = re.search("(.*?)\n", stripped_line).group(0)
				imlist.append(impath)
			except Exception as e:
				pass
			if os.path.isdir(impath):
				pass

	return imlist

def strong_aug(p=0.9):
	return Compose([
		Flip(),
		OneOf([
	        IAAAdditiveGaussianNoise(),
	        GaussNoise()
	    ], p=0.2),
	    OneOf([
	        MotionBlur(p=0.2),
	        MedianBlur(blur_limit=3, p=0.1),
	        Blur(blur_limit=3, p=0.1),
	    ], p=0.2),
	    ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1, rotate_limit=5, p=0.3),
	    OneOf([
	        OpticalDistortion(p=0.3),
	        GridDistortion(p=0.1),
	        IAAPiecewiseAffine(p=0.3),
	    ], p=0.2),
	    OneOf([
	        CLAHE(clip_limit=2),
	        IAASharpen(),
	        IAAEmboss(),
	        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.),
	    ], p=0.3),
	    HueSaturationValue(p=0.3),
	], p=p)

def simple_augmentation(img):
	trafo1 = strong_aug(p=0.9)

	try:
		return trafo1(image=img)['image']
	# TODO more elegant -> I don't know why I need to do this fix
	except Exception as e:
		pass
	return img


class ImageFilelist(data.Dataset):

	def __init__(self, root, flist, transform=None, target_transform=None, augmentation=None,
			flist_reader=default_flist_reader, loader=scale_loader, enrich_factor=1):
		self.enrich_factor = enrich_factor
		self.root   = root
		self.imlist = flist_reader(flist)
		self.org_list = self.imlist
		if enrich_factor <= 1:
			self.imlist = self.imlist
		else:
			self.imlist = ImageFilelist.enrich_imlist(self.imlist, enrich_factor)
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader
		self.augmentation = strong_aug(p=0.9)

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root, impath), augmentation=self.augmentation)
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.imlist)

	@classmethod
	def enrich_imlist(cls, imlist, enrich_factor):

		enrich_label = 1

		enriched_list = []

		for image, im_label in imlist:

			if im_label != enrich_label:
				enriched_list.append((image, im_label))
			else:
				for i in range(enrich_factor):
					enriched_list.append((image, im_label))

		random.shuffle(enriched_list)

		return enriched_list