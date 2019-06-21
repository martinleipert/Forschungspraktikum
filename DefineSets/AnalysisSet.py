import numpy
import os
import json
import random
import shutil

"""
Martin Leipert
martin.leipert@fau.de

Skript to generate analysis data set
"""

# How many data to use for training vs validation
file_no = 30

# Directories where the data are stored
notarsurkunden_home = "/home/martin/Forschungspraktikum/Testdaten/Notarsurkunden/notarsurkunden_mom/"

home = "/home/martin/Forschungspraktikum/Testdaten/"
dir_name = "analysis"

analysis_dir = os.path.join(home, dir_name)

if not os.path.exists(analysis_dir):
	os.mkdir(analysis_dir)
else:
	for filename in os.listdir(analysis_dir):
		os.remove(os.path.join(analysis_dir, filename))

notarsurkunden_files = os.listdir(notarsurkunden_home)
random.shuffle(notarsurkunden_files)
notarsurkunden_files = notarsurkunden_files[0:file_no]

for number, file in enumerate(notarsurkunden_files):
	src = os.path.join(notarsurkunden_home, file)
	dest = os.path.join(analysis_dir, "%i_%s" % (number, file))

	shutil.copy(src, dest)
