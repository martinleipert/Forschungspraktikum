import os
from PIL import Image

DIR_KEINE_NOTARSURKUNDEN = "/home/martin/Forschungspraktikum/Testdaten/KeineNotarsurkunden/keinenotarsurkunden_mom/"
DIR_NOTARSURKUNDEN = "/home/martin/Forschungspraktikum/Testdaten/Notarsurkunden/notarsurkunden_mom/"


def main():
	# cache_directory(DIR_KEINE_NOTARSURKUNDEN)
	cache_directory(DIR_NOTARSURKUNDEN, width=1000)
	pass


def cache_directory(dir_name, cache_dir_name="cached", width=500):

	cache_dir = os.path.join(dir_name, cache_dir_name)

	if not os.path.exists(cache_dir):
		os.mkdir(cache_dir)

	jpg_images = os.listdir(dir_name)
	jpg_images = list(filter(lambda x: x.lower().endswith(".jpg") or x.lower().endswith(".jpeg"), jpg_images))

	for jpg_image in jpg_images:
		store_path = os.path.join(cache_dir, jpg_image)

		# Load Image
		img_path = os.path.join(dir_name, jpg_image)
		image = Image.open(img_path)
		try:
			image.load()
		except Exception as e:
			print(e.__str__())

		# Rescale
		height = (width / image.size[1]) * image.size[0]
		thumb_size = [height, width]
		image.thumbnail(thumb_size, Image.ANTIALIAS)

		# Save image
		image.save(store_path)
	pass


if __name__ == "__main__":
	main()
