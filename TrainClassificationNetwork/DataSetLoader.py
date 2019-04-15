"""
Source: https://github.com/pytorch/vision/issues/81
"""

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import os.path
import re

Image.LOAD_TRUNCATED_IMAGES = True

def default_loader(path):

	image = Image.open(path)

	for i in range(3):
		try:
			image.load()
		except Exception as e:
			pass

	return image.convert('RGB')

def scale_loader(path):

	image = default_loader(path)
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
				imlist.append( (impath, int(imlabel)) )
			except Exception as e:
				pass
			if os.path.isdir(impath):
				pass

	return imlist

class ImageFilelist(data.Dataset):

	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=scale_loader):
		self.root   = root
		self.imlist = flist_reader(flist)
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root,impath))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.imlist)