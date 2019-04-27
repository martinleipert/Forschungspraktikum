import numpy
import os
import json
import random

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
STORE_FILE = "mini_set"

# Tags for the Json Files
TAG_TRAINING_SET = "TRAINING_SET"
TAG_TEST_SET = "TEST_SET"
TAG_VALIDATION_SET = "VALIDATION_SET"
TAG_URKUNDEN = "URKUNDEN"
TAG_KEINE_URKUNDEN = "KEINE_URKUNDEN"

MODE = 1

REDUCE_SET = True
REDUCTION = 0.1
BALANCE_SET = False


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


# Main method which shuffles and splits the sets
def main():
	# Get the filelist from the directories
	set_notarsurkunden = listfiles_fullpath(NOTARSURKUNDEN_DIR)
	set_keine_urkunden = listfiles_fullpath(KEINEURKUNDEN_DIR)

	if REDUCE_SET:
		set_notarsurkunden, set_keine_urkunden = reduce_sets([set_notarsurkunden, set_keine_urkunden], REDUCTION)

	if BALANCE_SET:
		set_notarsurkunden, set_keine_urkunden = balance_sets([set_notarsurkunden, set_keine_urkunden])

	# How many data of the set to use for  testing
	PERCENTAGE_TEST = 0.25

	# Percentage of the data in training actually used for validation
	PERCENTAGE_VALIDATION = 1. / 3.

	# Get the sets
	training_urkunden, validation_urkunden, test_urkunden = \
		split_set(set_notarsurkunden, PERCENTAGE_TEST, PERCENTAGE_VALIDATION)
	(training_keine_urkunden, validation_keine_urkunden, test_keine_urkunden) = \
		split_set(set_keine_urkunden, PERCENTAGE_TEST, PERCENTAGE_VALIDATION)

	# Mode == 0 -> JSON
	if MODE == 0:

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
		storage_path = os.path.join(STORE_DIR, STORE_FILE + ".json")

		# Store in a json file
		with open(storage_path, "w") as openfile:
			json.dump(data_dict, openfile, indent=4, sort_keys=True)
		pass

	# Mode == 0 -> FILES
	elif MODE == 1:

		training_set = [
			(training_keine_urkunden, 0),
			(training_urkunden, 1)
		]
		store_list(os.path.join(STORE_DIR, STORE_FILE, "traindata.txt"), training_set)

		validation_set = [
			(validation_keine_urkunden, 0),
			(validation_urkunden, 1)
		]
		store_list(os.path.join(STORE_DIR, STORE_FILE, "validationdata.txt"), validation_set)

		test_set = [
			(test_keine_urkunden, 0),
			(test_urkunden, 1)
		]
		store_list(os.path.join(STORE_DIR, STORE_FILE, "testdata.txt"), test_set)


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
def listfiles_fullpath(directory):
	files = os.listdir(directory)
	files = [os.path.join(directory, file) for file in files]
	files = list(filter(lambda x: os.path.isfile(x), files))
	return files


def split_set(data_set, percentage_test, percentage_validation):

	# Shuffle the dataset to randomize
	random.shuffle(data_set)

	# region set size calculation
	# Count the files
	no_data = len(data_set)

	# calculate the sizes of the sets
	no_testset = int(numpy.floor(percentage_test * no_data))

	# Calculate how much data are left fore training and validation
	no_train_and_val = no_data - no_testset

	# Calculate how much data to use for validation and training
	no_validation = int(numpy.floor(no_train_and_val * percentage_validation))
	# endregion set size calculation

	# region indices
	# calculate the indices for set splitting
	end_idx_test = no_testset
	end_idx_val = no_testset + no_validation
	end_idx_train = no_data

	# endregion indices

	# region splitting
	# split the set
	test_data = data_set[0:end_idx_test]
	validation_data = data_set[end_idx_test:end_idx_val]
	training_data = data_set[end_idx_val:end_idx_train]

	# endregion splitting
	return training_data, validation_data, test_data


if __name__ == '__main__':
	main()
