import torch
from PIL import Image, ImageDraw
import os
import random
from TrainSegmentationNetwork.UNetLoader_dynamic import extract_bounding_box
from xml.etree import ElementTree as ET
import numpy as np

XML_DIR = "page"
IMAGE_DIR = "notarsurkunden_mom"
SCHEMA = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
REGION_TAG = "GraphicRegion"


class SymbolSwapper:
	def __init__(self, path_to_files, swap_or_add=0.5):
		self.__file_paths = list(map(lambda x: x[0], filter(lambda x: x[1] == 1, path_to_files)))
		self.__swap_or_add = swap_or_add
		pass

	def get_xml(self, image_path):
		image_dir = os.path.dirname(image_path)
		file_name = os.path.basename(image_path)
		file_name = file_name.rstrip('.jpg')
		file_name = "%s.xml" % file_name
		xml_path = os.path.join(image_dir, XML_DIR, file_name)
		return xml_path

	def load_and_swap_symbol(self, org_doc):

		only_add = random.randint(0, 100) > (100*self.__swap_or_add)

		org_xml = self.get_xml(org_doc)
		swap_doc, swap_xml = self.get_random_file(org_doc)

		large_im = Image.open(org_doc)
		large_dim = large_im.size

		im = Image.open(os.path.join(os.path.dirname(org_doc), "cached", os.path.basename(org_doc)))
		for i in range(3):
			try:
				im.load()
			except Exception as e:
				print(e.__str__())
		org_image = im.convert('RGB')

		org_im_scale = im.size[1] / large_dim[1]

		del im

		# Not every notary document contains a signature
		elem = None

		while elem is None:
			for i in range(3):
				try:
					xml_tree = ET.parse(swap_xml)
				except Exception as e:
					print(e.__str__())
			root = xml_tree.getroot()
			elem = root.find('*/{' + SCHEMA + '}' + REGION_TAG)
			if elem is None:
				swap_doc, swap_xml = self.get_random_file(org_doc)

		im = Image.open(swap_doc)

		org_dim_swap = im.size

		im = Image.open(os.path.join(os.path.dirname(swap_doc), "cached", os.path.basename(swap_doc)))

		for i in range(3):
			try:
				im.load()
			except Exception as e:
				print(e.__str__())

		swap_image = im.convert('RGB')

		swap_im_scale = im.size[1] / org_dim_swap[1]

		del im

		points = elem.find('{' + SCHEMA + '}' + 'Coords').get('points')

		fix_pairs = points.split(' ')
		fix_pts = tuple(map(lambda x: tuple(map(int, x.split(','))), fix_pairs))
		fix_pts = tuple(map(lambda x: (x[0]*swap_im_scale, x[1]*swap_im_scale), fix_pts))

		mask = Image.new('L', [swap_image.size[0], swap_image.size[1]], 0)
		ImageDraw.Draw(mask).polygon(fix_pts, fill=1, outline=1, )

		# Extract notary sign
		xbb, ybb, hbb, wbb = extract_bounding_box(fix_pts)
		cropped_sign = swap_image.crop((xbb, ybb, xbb+hbb, ybb+wbb))

		# Not every notary document contains a signature
		xml_tree = ET.parse(org_xml)
		root = xml_tree.getroot()
		elem = root.find('*/{' + SCHEMA + '}' + REGION_TAG)

		if elem is None or only_add is True:

			width, height = org_image.size
			xbb, ybb = (100 + random.randint(0, width-200), 100 + random.randint(0, height-200))
			org_image.paste(cropped_sign, (xbb, ybb))
			pass

		else:
			points = elem.find('{' + SCHEMA + '}' + 'Coords').get('points')
			fix_pts = tuple(map(lambda x: tuple(map(int, x.split(','))), points.split(' ')))
			fix_pts = tuple(map(lambda x: (x[0]*org_im_scale, x[1]*org_im_scale), fix_pts))

			mask = Image.new('L', [swap_image.size[0], swap_image.size[1]], 0)
			ImageDraw.Draw(mask).polygon(fix_pts, fill=1, outline=1, )

			# Extract notary sign
			xbb, ybb, hbb, wbb = tuple(map(int, extract_bounding_box(fix_pts)))

			cropped_sign = cropped_sign.resize((hbb, wbb))
			org_image.paste(cropped_sign, (xbb, ybb))

		return org_image

	# Helper Method for bounding box extraction
	def extract_bounding_box(self, pt_arr):

		pt_arr = np.array(pt_arr)
		x_min, y_min = np.min(pt_arr, 0)
		x_max, y_max = np.max(pt_arr, 0)

		bb_coordinates = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
		bb_parameters = (x_min, y_min, x_max - x_min, y_max - y_min)
		return bb_parameters

	def get_random_file(self, org_file):
		sample = org_file

		while sample == org_file:
			sample = random.choice(self.__file_paths)

		return sample, self.get_xml(sample)


if __name__ == '__main__':
	test_path = "/home/martin/Forschungspraktikum/Testdaten/Transkribierte_Notarsurkunden/" \
				"notarskurkunden_mom_restored/0547_de_bayhsta_ku-rohr_0327_14691118_2c180219-9671-4dda-80a8-fbc33230fd9d_r.jpg"

	notary_root = "/home/martin/Forschungspraktikum/Testdaten/Transkribierte_Notarsurkunden/" \
					"notarskurkunden_mom_restored/"
	notary_files = os.listdir(notary_root)

	notary_files = [(os.path.join(notary_root, notary_file), 1) for notary_file in notary_files]

	swapper = SymbolSwapper(notary_files)
	xml_test = swapper.get_xml(test_path)

	if os.path.exists(xml_test) and os.path.isfile(xml_test):
		print("XML exists and is valid")

	image = swapper.load_and_swap_symbol(test_path)

	from matplotlib import pyplot

	pyplot.imshow(np.array(image))
	pyplot.show()
	pass
