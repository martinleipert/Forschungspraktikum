import os


from xmlpage_segmenter import create_UNet_training_images


DIR_ROOT = "/home/martin/Forschungspraktikum/Testdaten/"

NOTARSURKUNDEN_DIR = f"{DIR_ROOT}/Transkribierte_Notarsurkunden/notarskurkunden_mom_restored/"
XMLFILE_DIR = f"{DIR_ROOT}/Transkribierte_Notarsurkunden/notarskurkunden_mom_restored/page"
OUT_DIR = f"{DIR_ROOT}/Transkribierte_Notarsurkunden/notarskurkunden_mom_restored/segmented"
LABEL_FILE = f"{DIR_ROOT}/Transkribierte_Notarsurkunden/notarskurkunden_mom_restored/segmented/labels.txt"

IMG_TYPE = '.jpg'


# Possibilities
# ['mask', 'image', 'snippet'],
OUTPUT = 'mask'

BINARIZE = True
REGION_TYPE = "GraphicRegion"

# File  to note unprocessed images
UNPROCESSED_FILES = f"{DIR_ROOT}/Transkribierte_Notarsurkunden/unprocessed.txt"
SIEGEL_FILES = f"{DIR_ROOT}/Transkribierte_Notarsurkunden/siegel_files.txt"


def main():

	dir_content = os.listdir(NOTARSURKUNDEN_DIR)
	dir_content = filter(lambda x: x.endswith(IMG_TYPE), dir_content)

	open_unprocessed_file = open(UNPROCESSED_FILES, "w+")
	siegel_file = open(SIEGEL_FILES, "w+")

	for image in dir_content:
		img_name = image.strip(IMG_TYPE)

		xml_file = f"{XMLFILE_DIR}/{img_name}.xml"
		img_path = f"{os.path.join(NOTARSURKUNDEN_DIR, image)}"

		try:
			mask_path = create_UNet_training_images(
				xml_file, REGION_TYPE, NOTARSURKUNDEN_DIR, IMG_TYPE, LABEL_FILE, OUT_DIR, OUTPUT, BINARIZE)
			siegel_file.write(f"{img_path}, {mask_path}\n")
		except Exception as e:
			open_unprocessed_file.write(img_path)

	open_unprocessed_file.close()
	siegel_file.close()
	pass

if __name__ == '__main__':
	main()
