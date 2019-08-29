"""
Source: https://github.com/pytorch/vision/issues/81
"""

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import os.path
import re
import random
from Augmentations.Augmentations import *
from TrainClassificationNetwork.SwapAddNotarySymbol import SymbolSwapper
import torch
import numpy

Image.LOAD_TRUNCATED_IMAGES = True


def augmentation_loader(path, label, augmentation=None, swapper=None):

	if (swapper is not None) and (label == 1) and (random.randint(1, 10) > 1):
		image = swapper.load_and_swap_symbol(path)

	else:
		img_path = os.path.join(os.path.dirname(path), "cached", os.path.basename(path))

		image = Image.open(img_path)

		for i in range(3):
			try:
				image.load()
			except Exception as e:
				print(e.__str__())

		image = image.convert('RGB')

	if augmentation:

		trans0 = transforms.Resize(512)
		image = trans0(image)
		"""
		# Shrink the image for augmentation -> Save computation time
		factor = 512.0 / min(image.size)
		thumb_size = list(map(lambda x: factor*x, image.size))
		image.thumbnail(thumb_size, Image.ANTIALIAS)
		"""
		try:
			image = numpy.array(image)
			augmented_image = augmentation(image=image)
			image = augmented_image["image"]
			image = Image.fromarray(image)
		except Exception as e:
			print(e.__str__())

	trans1 = transforms.Resize(224)
	trans2 = transforms.CenterCrop(224)
	trans3 = transforms.ToTensor()
	image = trans3(trans2(trans1(image)))

	return image.detach()


# Read the previously defined filelist (from Define Sets)
def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n ...(same to caffe's filelist)
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			try:
				stripped_line = line.strip()
				im_path, im_label = re.search(r"(.*?)\s+(\d)", stripped_line).groups()
				if os.path.isdir(im_path):
					pass
				imlist.append((im_path, int(im_label)))

			except Exception as e:
				pass

	return imlist


class ImageFileList(data.Dataset):

	def __init__(self, root, flist, transform=None, augmentation=weak_augmentation,
				flist_reader=default_flist_reader, loader=augmentation_loader, enrich_factor=1, swap=False,
				swap_probability=0.9, swap_or_add=0.5):
		self.enrich_factor = enrich_factor
		self.root = root
		self.im_list = flist_reader(flist)
		self.org_list = self.im_list
		if enrich_factor <= 1:
			self.im_list = self.im_list
		else:
			self.im_list = ImageFileList.augment_imlist(self.im_list, enrich_factor)
		self.transform = transform
		self.loader = loader
		if augmentation is not None:
			self.augmentation = augmentation()
		else:
			self.augmentation = None

		self.swap_probability = swap_probability
		self.swapper = None
		if swap is True:
			self.swapper = SymbolSwapper(self.org_list, swap_or_add)

	def __getitem__(self, index):
		impath, target = self.im_list[index]
		img = self.loader(os.path.join(self.root, impath), target, augmentation=self.augmentation, swapper=self.swapper)
		if self.transform is not None:
			img = self.transform(img)

		return img, target, impath

	def __len__(self):
		return len(self.im_list)

	@classmethod
	def augment_imlist(cls, im_list, enrich_factor):

		enrich_label = 1

		enriched_list = []

		for image, im_label in im_list:

			if im_label != enrich_label:
				enriched_list.append((image, im_label))
			else:
				for i in range(enrich_factor):
					enriched_list.append((image, im_label))

		random.shuffle(enriched_list)

		return enriched_list


if __name__ == '__main__':
	path = "/home/martin/Forschungspraktikum/Testdaten/Notarsurkunden/notarsurkunden_mom/0032_AES_14640607_ReiheD-340_r.jpg"
	image = augmentation_loader(path, 1, augmentation=weak_augmentation())
	image = image.cpu().numpy()

	from matplotlib import pyplot

	img = numpy.zeros([224, 224, 3])
	img[:, :, 0] = image[0, :, :]
	img[:, :, 1] = image[1, :, :]
	img[:, :, 2] = image[2, :, :]

	pyplot.imshow(img)
	pyplot.show()
	pass
