import os
from PIL import Image

KEINEURKUNDEN_DIR = "/home/martin/Forschungspraktikum/Testdaten/KeineNotarsurkunden/keinenotarsurkunden_mom/"

os.chdir(KEINEURKUNDEN_DIR)


for counter, file in enumerate(os.listdir(".")):

	if counter % 100 == 0:
		print(counter)

	if not os.path.isfile(file):
		continue

	try:
		opened = Image.open(file)
		opened.close()
	except Exception as e:
		print("Could not load %s" % file)

		with open(file, "w+b") as openfile:
			openfile.write(openfile.read())

	pass
