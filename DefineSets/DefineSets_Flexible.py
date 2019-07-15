import numpy
import os
import json
import random

from argparse import ArgumentParser

"""
Martin Leipert
martin.leipert@fau.de

Skript to generate test data sets
"""

# Directories where the data are stored
NOTARSURKUNDEN_DIR = "/home/martin/Forschungspraktikum/Testdaten/Notarsurkunden/notarsurkunden_mom/"
KEINEURKUNDEN_DIR = "/home/martin/Forschungspraktikum/Testdaten/KeineNotarsurkunden/keinenotarsurkunden_mom/"

# Where the defined sets are stored
STORE_DIR = "/home/martin/Forschungspraktikum/Testdaten/Sets/"


# Tags for the Json Files
TAG_TRAINING_SET = "TRAINING_SET"
TAG_TEST_SET = "TEST_SET"
TAG_VALIDATION_SET = "VALIDATION_SET"

TAG_URKUNDEN = "URKUNDEN"
TAG_KEINE_URKUNDEN = "KEINE_URKUNDEN"


# Main method which shuffles and splits the sets
def main():
	arg_parser = ArgumentParser("Parametrizable set creator")
	arg_parser.add_argument("SET_NAME", help="name of the set")
	arg_parser.add_argument("PERCENTAGE_TEST", type=float, help="How many files will be used for testing")
	arg_parser.add_argument("PERCENTAGE_VALIDATION", type=float, 
						help="How many files of the remaining will be used for validation")
	arg_parser.add_argument("--storeMode", type=str, default="TEXT",
							help="How to store as json or txt. Possible Inputs: 'TEXT', 'JSON'")
	arg_parser.add_argument("--reduceByPercent", type=float, default=None,
							help="Give a percentage to which the set will be reduced. No reduction if not given")
	arg_parser.add_argument("--balanceSet", type=bool, default=False,
							help="Use as many notary documents as non-notary in the set. "
								"Of course it only works for small sets")

	parsed_args = arg_parser.parse_args()

	set_name = parsed_args.SET_NAME
	# How many data of the set to use for  testing
	percentage_test = parsed_args.PERCENTAGE_TEST
	# Percentage of the data in training actually used for validation
	percentage_validation = parsed_args.PERCENTAGE_VALIDATION

	store_mode = parsed_args.storeMode

	reduce_set = False if parsed_args.reduceByPercent is None else True
	reduction = parsed_args.reduceByPercent
	balance_set = parsed_args.balanceSet

	# Get the filelist from the directories
	set_notarsurkunden = list_files_full_path(NOTARSURKUNDEN_DIR)
	set_keine_urkunden = list_files_full_path(KEINEURKUNDEN_DIR)

	if reduce_set:
		set_notarsurkunden, set_keine_urkunden = reduce_sets([set_notarsurkunden, set_keine_urkunden], reduction)

	if balance_set:
		set_notarsurkunden, set_keine_urkunden = balance_sets([set_notarsurkunden, set_keine_urkunden])

	# Get the sets
	# Notary documents
	training_urkunden, validation_urkunden, test_urkunden = \
		split_set(set_notarsurkunden, percentage_test, percentage_validation)

	# Not notary documents
	(training_keine_urkunden, validation_keine_urkunden, test_keine_urkunden) = \
		split_set(set_keine_urkunden, percentage_test, percentage_validation)

	# Mode == 0 -> JSON
	if store_mode == "JSON":

		# Store into a dictionary
		data_dict = dict()
		data_dict[TAG_TRAINING_SET] = {
			TAG_URKUNDEN: training_urkunden,
			TAG_KEINE_URKUNDEN: training_keine_urkunden,
		}
		data_dict[TAG_VALIDATION_SET] = {
			TAG_URKUNDEN: validation_urkunden,
			TAG_KEINE_URKUNDEN: validation_keine_urkunden
		}
		data_dict[TAG_TEST_SET] = {
			TAG_URKUNDEN: test_urkunden,
			TAG_KEINE_URKUNDEN: test_keine_urkunden,
		}

		# Combine the path to store in
		storage_path = os.path.join(STORE_DIR, set_name + ".json")

		# Store in a json file
		with open(storage_path, "w") as openfile:
			json.dump(data_dict, openfile, indent=4, sort_keys=True)
		pass

	# Mode == 0 -> FILES
	elif store_mode == "TEXT":

		training_set = [
			(training_keine_urkunden, 0),
			(training_urkunden, 1)
		]
		store_list(os.path.join(STORE_DIR, set_name, "traindata.txt"), training_set)

		validation_set = [
			(validation_keine_urkunden, 0),
			(validation_urkunden, 1)
		]
		store_list(os.path.join(STORE_DIR, set_name, "validationdata.txt"), validation_set)

		test_set = [
			(test_keine_urkunden, 0),
			(test_urkunden, 1)
		]
		store_list(os.path.join(STORE_DIR, set_name, "testdata.txt"), test_set)


# Store as file list in plain text
def store_list(outputfile, data_label_list):
	outputdir = os.path.dirname(outputfile)
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)

	lines = ""
	shuffled_list = []

	for data_list, label in data_label_list:
		for data_file in data_list:
			shuffled_list.append((data_file, label))

	random.shuffle(shuffled_list)

	for (data_file, label) in shuffled_list:
			lines += "%s %i\n" % (data_file, label)

	with open(outputfile, "w+") as openfile:
		openfile.write(lines)


# Method to generate the filepaths for a directory
def list_files_full_path(directory):
	files = os.listdir(directory)
	files = [os.path.join(directory, file) for file in files]
	files = list(filter(lambda x: os.path.isfile(x) and x.lower().endswith('jpg'), files))
	return files


def split_set(data_set, percentage_test, percentage_validation):

	# Shuffle the dataset to randomize
	random.shuffle(data_set)

	# Count the files
	no_data = len(data_set)
	# calculate the sizes of the sets
	no_testset = int(numpy.floor(percentage_test * no_data))
	# Calculate how much data are left fore training and validation
	no_train_and_val = no_data - no_testset
	# Calculate how much data to use for validation and training
	no_validation = int(numpy.floor(no_train_and_val * percentage_validation))

	# calculate the indices for set splitting
	end_idx_test = no_testset
	end_idx_val = no_testset + no_validation
	end_idx_train = no_data

	# split the set
	test_data = data_set[0:end_idx_test]
	validation_data = data_set[end_idx_test:end_idx_val]
	training_data = data_set[end_idx_val:end_idx_train]

	return training_data, validation_data, test_data


#  region Helpers for Class Distribution
# To equalize the nr of samples of each clas
def balance_sets(data_sets):
	len_list = list()

	for data_set in data_sets:
		random.shuffle(data_set)
		len_list.append(len(data_set))

	smallest = min(len_list)

	new_sets = list()
	for data_set in data_sets:
		new_sets.append(data_set[0:smallest])

	return new_sets


def reduce_sets(data_sets, factor):
	new_sets = []
	for data_set in data_sets:
		new_last_idx = int(numpy.ceil(len(data_set)*factor))
		new_sets.append(data_set[0:new_last_idx])

	return new_sets
# endregion Helpers for Class Distribution


if __name__ == '__main__':
	main()
