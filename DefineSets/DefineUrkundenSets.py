import os
import random
import numpy
from xml.etree import ElementTree as ET


"""
Define the Notarsurkunden Sets for the UNet Training

"""

SCHEMA = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

REGION_TYPES = {
    "Background": 0,
    "TextRegion": 1,
    "ImageRegion": 2,
    "GraphicRegion": 3
}

# How many data of the set to use for  testing
PERCENTAGE_TEST = 0.1
# Percentage of the data in training actually used for validation
PERCENTAGE_VALIDATION = 1. / 9.

NOTARSURKUNDEN_SETS = "/home/martin/Forschungspraktikum/Testdaten/Segmentation_Sets/"
NOTARSURKUNDEN_DIR = "/home/martin/Forschungspraktikum/Testdaten/Transkribierte_Notarsurkunden/" \
                     "notarskurkunden_mom_restored/"

REDUCE_SET = False
REDUCTION = 0.5

SET_NAME = "full_set"


def main():
	file_list = []

	ns = '{' + SCHEMA + '}'

	for file in os.listdir(NOTARSURKUNDEN_DIR):
		file_path = os.path.join(NOTARSURKUNDEN_DIR, file)
		if os.path.isfile(file_path):
			if not file.lower().endswith('.jpg'):
				continue
			dir_name = os.path.dirname(file_path)
			filename = os.path.basename(file_path).rsplit('.jpg')[0]
			xml_filename = f"{filename}.xml"
			xml_path = os.path.join(dir_name, 'page', xml_filename)

			xml_tree = ET.parse(xml_path)

			root = xml_tree.getroot()

			el_list = []

			for key in REGION_TYPES.keys():
				els = root.findall("*/" + ns + key)
				el_list.extend(els)

			if len(el_list) == 0:
				continue

			file_list.append(file_path)

	# Shuffle the list to just split via indices
	random.shuffle(file_list)

	# Calculate the indices
	nr_idx = len(file_list)

	if REDUCE_SET:
		nr_idx = nr_idx*REDUCTION

	perc_trainval = (1 - PERCENTAGE_TEST)
	train_idx = numpy.round(nr_idx * perc_trainval * (1 - PERCENTAGE_VALIDATION))
	val_idx = train_idx + numpy.round(nr_idx * perc_trainval * PERCENTAGE_VALIDATION)

	train_set = file_list[0:int(train_idx-1)]
	val_set = file_list[int(train_idx):int(val_idx-1)]
	test_set = file_list[int(val_idx):int(numpy.round(nr_idx))]

	store_dir = os.path.join(NOTARSURKUNDEN_SETS, SET_NAME)

	if not os.path.exists(store_dir):
		os.mkdir(store_dir)

	train_file = os.path.join(store_dir, "training.txt")
	val_file = os.path.join(store_dir, "validation.txt")
	test_file = os.path.join(store_dir, "test.txt")

	store_list(train_file, train_set)
	store_list(val_file, val_set)
	store_list(test_file, test_set)
	pass


def store_list(file_path, file_list):

	with open(file_path, "w+") as openfile:

		for file_of_set in file_list:
			openfile.write("%s\n" % file_of_set)


if __name__ == '__main__':
	main()
