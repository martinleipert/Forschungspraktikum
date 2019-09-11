import torch
from TrainSegmentationNetwork.Train_UNet import calc_loss
from TrainSegmentationNetwork.UNetLoader_dynamic import UNetDatasetDynamicMask
from UNet.PlotSegmentationResults import plot_result
from UNet.BatchNormUNet import UNet
from argparse import ArgumentParser
import re
from collections import defaultdict
import numpy as np
import skimage
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.morphology as morph
from PIL import Image
from TrainSegmentationNetwork.IntersectionOverUnion import calculateIoU

"""
Martin Leipert
martin.leipert@fau.de

"""


BATCH_SIZE = 5

NUM_CLASSES = 4

TEST_LOSSES_WEIGHTING = {
	"BCE_LOSS": 1,
	"DICE_LOSS": 1,
	"FOCAL_LOSS": 1
}


def main():
	arg_parser = ArgumentParser("Test the Unet on the defined test data")
	arg_parser.add_argument("model_name", help="pth file to the model")

	parsed_args = arg_parser.parse_args()

	model_name = parsed_args.model_name

	set_name = "fullset"

	file_list_test = "/home/martin/Forschungspraktikum/Testdaten/Segmentation_Sets/%s/test.txt" % set_name

	test_data = UNetDatasetDynamicMask(file_list_test, region_select=False)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

	# Evaluate on CUDA if possible
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_network = UNet(NUM_CLASSES)
	model_network.load_state_dict(torch.load("TrainedModels/%s" % model_name))
	model_network.to(device)

	base_name = model_name.rstrip(".pth")

	metrics = defaultdict(float)
	# {'focal': 0, 'dice': 0, 'bce': 0, 'loss': 0}

	model_network.eval()
	torch.no_grad()


	loss_sum = 0

	confusion = np.zeros([4, 2])

	ground_truth = np.zeros([4])

	iou = np.zeros((4, 1))

	# Testings
	for images, masks, image_paths in test_loader:
		images = images.to(device)
		masks = masks.to(device)

		# forward
		# track history if only in train
		outputs = model_network(images)
		outputs = outputs.detach()
		masks = masks.detach()
		images = images.detach()

		loss = calc_loss(outputs, masks, metrics, TEST_LOSSES_WEIGHTING)

		# Update the losses
		loss_sum += loss / len(images)

		# plot_result(outputs, images, base_name, image_paths)

		this_confusion, gt, this_iou = get_segmented_area(outputs, masks, images, image_paths)
		confusion = np.add(confusion, this_confusion)
		ground_truth = np.add(ground_truth, gt)
		iou += this_iou

	iou = iou / len(test_loader.dataset.input_images)
	confusion = confusion / len(test_loader.dataset.input_images)
	ground_truth = ground_truth / len(test_loader.dataset.input_images)
	print("Ground truth")
	print(ground_truth)

	print(f"Overall loss {loss_sum.cpu().item()}")

	print("IOU: %.5f\t\t%.5f\t\t%.5f\t\t%.5f\n" % tuple(iou[:, 0]))
	denote_result(base_name, loss_sum, metrics, confusion, iou)


def get_segmented_area(prediction, org_mask, raw_images, image_paths):

	prediction = torch.sigmoid(prediction.double())
	prediction = prediction.cpu().numpy()

	org_mask = org_mask.detach().cpu().numpy()
	raw_images = raw_images.detach().cpu().numpy()

	iou = calculateIoU(prediction, org_mask)

	class_labels = np.argmax(prediction, axis=1)

	image_size = 224*224

	full_confusion = np.zeros([4, 2])
	ground_truth = np.zeros(4)

	for i in range(prediction.shape[0]):

		local_mask = org_mask[i, :, :, :]
		mask_labels = np.argmax(local_mask, axis=0)
		local_labels = class_labels[i, :, :]

		image = raw_images[i, :, :]
		raw_im = np.zeros(list(image.shape[1:3]) + [3])

		for i in range(3):
			raw_im[:, :, i] = image[i, :, :]

		image = Image.fromarray(np.uint8(raw_im*256))
		image = image.convert("L")
		thresh = filters.threshold_otsu(np.array(image))
		text_or_sign = image < thresh
		background = image > thresh

		confusion = np.zeros([4, 2])
		for i in range(4):
			ground_truth[i] += np.where(mask_labels == i, 1, 0).sum(0).sum(0) / image_size

			correct = np.where(np.logical_and(local_labels == i, mask_labels == i), 1, 0).sum(0).sum(0) / image_size
			false = np.where(np.logical_and(local_labels == i, mask_labels != i), 1, 0).sum(0).sum(0) / image_size

			confusion[i, 0] = correct
			confusion[i, 1] = false

			pass
		full_confusion = np.add(full_confusion, confusion)

	"""
	# Missclassified area:
	subtraction = new_mask - org_mask
	subtraction = np.where(subtraction < 0, 0, subtraction)
	missclassified = subtraction.sum(axis=0).sum(axis=0)

	org_images = []

	for raw_im in raw_images:

		image = Image.open(org_im)
		image = image.convert("L")

		# Correctly classified area:
		# Use Otsu thresholding

		image = Image.fromarray(raw_im)

		segmented = filters.threshold_otsu(image)
		morph.binary_dilation(segmented, out=segmented)
		org_images.append(segmented)
		"""
	return full_confusion, ground_truth, iou


def denote_result(base_name, loss, metrics, confusion, iou):

	with open(f"Results/{base_name}_test_result.txt", "w+") as open_file:
		open_file.write(f"Loss on Test_data{loss}\n")
		open_file.write(f"BCE Loss on Test_data{metrics['BCE_LOSS']}\n")
		open_file.write(f"Dice Loss on Test_data{metrics['DICE_LOSS']}\n")
		open_file.write(f"Focal Loss on Test_data{metrics['FOCAL_LOSS']}\n")
		open_file.write("\n")
		open_file.write("----- Confusion: -----\n")
		for i in range(4):
			open_file.write("%i: %.5f | %.5f \n" % (i, confusion[i, 0], confusion[i, 1]))

		open_file.write("\n")
		open_file.write("IOU: %.5f\t\t%.5f\t\t%.5f\t\t%.5f\n" % tuple(iou[:, 0]))


if __name__ == '__main__':
	main()
