import torch
from TrainSegmentationNetwork.Train_UNet import calc_loss
from TrainSegmentationNetwork.UNetLoader_dynamic import UNetDatasetDynamicMask
from UNet.PlotSegmentationResults import plot_result
from UNet.BatchNormUNet import UNet
from argparse import ArgumentParser
import re
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

	set_name = re.search(r"unet_(.*?_set)", model_name).group(1)

	file_list_test = "/home/martin/Forschungspraktikum/Testdaten/Segmentation_Sets/%s/test.txt" % set_name

	test_data = UNetDatasetDynamicMask(file_list_test, region_select=False)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

	# Evaluate on CUDA if possible
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_network = UNet(NUM_CLASSES)
	model_network.load_state_dict(torch.load("TrainedModels/%s" % model_name))
	model_network.to(device)

	base_name = model_name.rstrip(".pth")

	metrics = {'focal': 0, 'dice': 0, 'bce': 0, 'loss': 0}

	model_network.eval()
	torch.no_grad()

	loss_sum = 0
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

		plot_result(outputs, images, base_name, image_paths)

	print(f"Overall loss {loss_sum}")
	denote_result(base_name, loss_sum)


def denote_result(base_name, loss, metrics):

	with open(f"Results/{base_name}_test_result.txt", "w+") as open_file:
		open_file.write(f"Loss on Test_data{loss}\n")
		open_file.write(f"BCE Loss on Test_data{metrics['BCE_LOSS']}\n")
		open_file.write(f"Dice Loss on Test_data{metrics['DICE_LOSS']}\n")
		open_file.write(f"Focal Loss on Test_data{metrics['FOCAL_LOSS']}\n")


if __name__ == '__main__':
	main()
