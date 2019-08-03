import os
import random
import numpy
from xml.etree import ElementTree as ET
from argparse import ArgumentParser

"""
Define the Notarsurkunden Sets for the UNet Training

Sets are stored in directories which contain fies listing training, validation and test set
"""

SCHEMA = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

REGION_TYPES = {
	"Background": 0,
	"TextRegion": 1,
	"ImageRegion": 2,
	"GraphicRegion": 3
}

SET_ROOT = "/home/martin/Forschungspraktikum/Testdaten"
NOTARSURKUNDEN_SETS = f"{SET_ROOT}/Segmentation_Sets/"
NOTARSURKUNDEN_DIR = f"{SET_ROOT}/Transkribierte_Notarsurkunden/notarskurkunden_mom_restored/"


def main():
	arg_parser = ArgumentParser("Define sets for the Training of the Segmentation Network")
	arg_parser.add_argument("SET_NAME", type=str, help="Desired name of the set")
	arg_parser.add_argument("PERCENTAGE_TEST", type=float, help="How many percent to use for testing")
	arg_parser.add_argument("PERCENTAGE_VALIDATION", type=float,
							help="How many percent of the remaining are used for validation")
	arg_parser.add_argument("--sizeReduction", type=float, default=None, help="Reduce the set by how many percent")

	parsed_args = arg_parser.parse_args()

	reduce_set = False if parsed_args.sizeReduction is None else True
	reduction = parsed_args.sizeReduction

	set_name = parsed_args.SET_NAME

	# How many data of the set to use for  testing
	percentage_test = parsed_args.PERCENTAGE_TEST
	# Percentage of the data in training actually used for validation
	percentage_validation = parsed_args.PERCENTAGE_VALIDATION

	file_list = []

	ns = '{' + SCHEMA + '}'

	for file in os.listdir(NOTARSURKUNDEN_DIR):
		file_path = os.path.join(NOTARSURKUNDEN_DIR, file)

		# Extract the image filename and check if it contains class assignments
		if os.path.isfile(file_path):
			if not file.lower().endswith('.jpg'):
				continue

			# Get the xml file
			dir_name = os.path.dirname(file_path)
			filename = os.path.basename(file_path).rsplit('.jpg')[0]
			xml_filename = f"{filename}.xml"
			xml_path = os.path.join(dir_name, 'page', xml_filename)

			# Parse the xml
			xml_tree = ET.parse(xml_path)
			root = xml_tree.getroot()

			# Filter unlabeled files
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

	if reduce_set:
		nr_idx = nr_idx*reduction

	# Calculate the indices
	perc_trainval = (1 - percentage_test)
	train_idx = numpy.round(nr_idx * perc_trainval * (1 - percentage_validation))
	val_idx = train_idx + numpy.round(nr_idx * perc_trainval * percentage_validation)

	# Compose the sets
	train_set = file_list[0:int(train_idx-1)]
	val_set = file_list[int(train_idx):int(val_idx-1)]
	test_set = file_list[int(val_idx):int(numpy.round(nr_idx))]

	store_dir = os.path.join(NOTARSURKUNDEN_SETS, set_name)

	if not os.path.exists(store_dir):
		os.mkdir(store_dir)

	# Write into a File
	train_file = os.path.join(store_dir, "training.txt")
	store_list(train_file, train_set)
	val_file = os.path.join(store_dir, "validation.txt")
	store_list(val_file, val_set)
	test_file = os.path.join(store_dir, "test.txt")
	store_list(test_file, test_set)
	pass


def store_list(file_path, file_list):
	with open(file_path, "w+") as openfile:
		for file_of_set in file_list:
			openfile.write("%s\n" % file_of_set)


if __name__ == '__main__':
	main()
