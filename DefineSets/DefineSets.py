import numpy
import os
import json
import random

"""
Martin Leipert
martin.leipert@fau.de

Skript to generate test data sets
"""

# How many data to use for training vs validation
PERCENTAGE_TRAINING = 0.8
PERCENTAGE_VALIDATION = 1.0 - PERCENTAGE_TRAINING

# Directories where the data are stored
NOTARSURKUNDEN_DIR = "/home/martin/Forschungspraktikum/Testdaten/Notarsurkunden/notarsurkunden_mom/"
KEINEURKUNDEN_DIR = "/home/martin/Forschungspraktikum/Testdaten/KeineNotarsurkunden/keinenotarsurkunden_mom/"

# Where the defined sets are stored
STORE_DIR = "/home/martin/Forschungspraktikum/Testdaten/Sets/"
STORE_FILE = "first_set"

# Tags for the Json Files
TAG_TRAINING_SET = "TRAINING_SET"
TAG_TEST_SET = "TEST_SET"
TAG_URKUNDEN = "URKUNDEN"
TAG_KEINE_URKUNDEN = "KEINE_URKUNDEN"

MODE = 1


# Main method which shuffles and splits the sets
def main():
	# Get the filelist from the directories
	notarsurkunden = listfiles_fullpath(NOTARSURKUNDEN_DIR)
	keine_urkunden = listfiles_fullpath(KEINEURKUNDEN_DIR)

	# Count the files
	no_urkunden = len(notarsurkunden)
	no_keine_urkunden = len(keine_urkunden)

	# Shuffle the lists
	random.shuffle(notarsurkunden)
	random.shuffle(keine_urkunden)

	# Calculate the index to split
	end_idx_testset_notarsurkunden = int(numpy.ceil(PERCENTAGE_VALIDATION * no_urkunden))
	end_idx_testset_keine_urkunden = int(numpy.ceil(PERCENTAGE_VALIDATION * no_keine_urkunden))

	# The first files belong to the testset
	test_urkunden = notarsurkunden[0:end_idx_testset_notarsurkunden]
	test_keine_urkunden = keine_urkunden[0:end_idx_testset_keine_urkunden]

	# The rest is the trainingset
	training_urkunden = notarsurkunden[end_idx_testset_notarsurkunden:-1]
	training_keine_urkunden = keine_urkunden[end_idx_testset_keine_urkunden:-1]

	# Mode == 0 -> JSON
	if MODE == 0:

		# Store into a dictionary
		data_dict = dict()
		data_dict[TAG_TRAINING_SET] = {
			TAG_URKUNDEN: training_urkunden,
			TAG_KEINE_URKUNDEN: training_keine_urkunden,
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
	for data_list, label in data_label_list:
		for data_file in data_list:
			lines += "%s %i\n" % (data_file, label)

	with open(outputfile, "w+") as openfile:
		openfile.write(lines)


# Method to generate the filepaths for a directory
def listfiles_fullpath(directory):
	files = os.listdir(directory)
	files = [os.path.join(directory, file) for file in files]
	return files


if __name__ == '__main__':
	main()
