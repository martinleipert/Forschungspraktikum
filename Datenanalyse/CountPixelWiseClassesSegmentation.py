from TrainSegmentationNetwork.UNetLoader_dynamic import UNetDatasetDynamicMask
import torch
import numpy as np

BATCH_SIZE = 128

FILE_LIST = "/home/martin/Forschungspraktikum/Testdaten/Segmentation_Sets/all_files"


def main():

	data = UNetDatasetDynamicMask(FILE_LIST, region_select=False)
	loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE)

	class_count = np.zeros((1, 4))

	# Testings
	for images, masks, image_paths in loader:
		class_count += masks.sum([0, 2, 3]).cpu().detach().numpy()

	weights = np.divide(np.ones(np.shape(class_count)), 4*np.divide(class_count, class_count.sum()))

	print(f"Weights {weights[0, 0]}, {weights[0, 1]}, {weights[0, 2]}, {weights[0, 3]}")

	with open("calculated_weights.txt", "w") as openfile:
		openfile.write(f"Weights {weights[0, 0]}, {weights[0, 1]}, {weights[0, 2]}, {weights[0, 3]}")


if __name__ == "__main__":
	main()
