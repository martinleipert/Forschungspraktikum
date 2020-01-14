import torch
from torch import nn
from torchvision import models
import numpy as np
from TrainClassificationNetwork.DataSetLoader import ImageFileList
import os
from argparse import ArgumentParser
import re

"""
Martin Leipert
martin.leipert@fau.de

Test script for the trained classification network

Tests sensitivity and specifity 
Denotes the results as txt
Denote missclassifications
"""


BATCH_SIZE = 20
NUM_CLASSES = 2


def main():
	arg_parser = ArgumentParser("Evaluate a model")
	arg_parser.add_argument("model_name", help="name of the model")
	arg_parser.add_argument("batch_size", type=int)


	parsed_args = arg_parser.parse_args()
	model_name = parsed_args.model_name
	BATCH_SIZE = parsed_args.batch_size

	model_type = re.search("((?:resnet18)|(?:resnet50)|(?:densenet121))", model_name.lower()).group(1)
	model_type = model_type.upper()

	set_name = "full_set" # re.search("^(.*?_set)", model_name).groups(1)

	file_list_test = "/home/martin/Forschungspraktikum/Testdaten/Sets/%s/testdata.txt" % set_name

	test_data = ImageFileList(".", file_list_test)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, drop_last=False)

	# Evaluate on CUDA if possible
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if model_type == "RESNET18":
		model_network = models.resnet18()
		model_network.fc = nn.Sequential(
			nn.Linear(512, BATCH_SIZE), nn.ReLU(), nn.Dropout(0.2), nn.Linear(BATCH_SIZE, 2), nn.LogSoftmax(dim=1))

	elif model_type == "RESNET50":
		model_network = models.resnet50()
		model_network.fc = nn.Sequential(
			nn.Linear(2048, BATCH_SIZE), nn.ReLU(), nn.Dropout(0.2), nn.Linear(BATCH_SIZE, 2), nn.LogSoftmax(dim=1))

	elif model_type == "DENSENET121":
		# Pick the model
		model_network = models.densenet121()
		# -> Map the 1000 outputs to the 2 class problem
		model_network.classifier = nn.Linear(1024, NUM_CLASSES)

	state_dict = torch.load("TrainedModels/%s" % model_name)
	model_network.load_state_dict(state_dict)
	model_network.to("cuda:0")

	# Loss functions for testing
	nn_loss_fct = torch.nn.NLLLoss().to(device)
	ce_loss_fct = torch.nn.CrossEntropyLoss().to(device)

	# Running variables
	loss_sum_nn = 0
	loss_sum_ce = 0

	accuracy = 0
	confusion = np.zeros((2, 2))

	for_label_result_printing = []

	for param in model_network.parameters():
		param.requires_grad = False

	model_network.eval()
	# Training
	for images, labels, image_paths in test_loader:

		images = images.to(device)
		labels = labels.to(device)
		# forward
		# track history if only in train
		logps = model_network(images)

		batch_loss_nn = nn_loss_fct(logps, labels)
		batch_loss_ce = ce_loss_fct(logps, labels)

		ps = torch.exp(logps)
		top_p, top_class = ps.topk(1, dim=1)
		equals = top_class == labels.view(*top_class.shape)
		accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

		# Get the confusion matrix
		# Transform to CPU
		ref_labels = np.array(labels.view(*top_class.shape).cpu())
		pred_label = np.array(top_class.cpu())

		for index, im_path in enumerate(image_paths):
			im_base_name = os.path.basename(im_path)
			to_print = (im_base_name, label_to_str(ref_labels[index]), label_to_str(pred_label[index]))
			for_label_result_printing.append(to_print)

		# Get the confusion matrix by a queue
		# 1,1 = Correctly classified notary documents
		for i in range(NUM_CLASSES):
			for j in range(NUM_CLASSES):
				match = np.logical_and(ref_labels == i, pred_label == j)
				confusion[i, j] += list(match).count(True)

		# Update the losses
		loss_sum_nn += batch_loss_nn.item() / len(test_loader)
		loss_sum_ce += batch_loss_ce.item() / len(test_loader)

		del images, labels, logps, ps
		torch.cuda.empty_cache()

	print(f"Overall loss {loss_sum_nn}")
	denote_result(model_name, loss_sum_ce, loss_sum_nn, confusion, for_label_result_printing)


def denote_result(base_name, ce_loss, nn_loss, confusion, for_label_result_printing):

	with open(f"Results/{base_name}_test_result.txt", "w+") as open_file:
		open_file.write(f"Cross Entropy: {ce_loss}\n")
		open_file.write(f"NN Loss: {nn_loss}\n")
		open_file.write("\n")

		# Pretty-print the confusion table
		open_file.write("Doc Type   | Correctly classified | Missclassified\n" + "-" * 50 + "\n" +
						"Non-notary | %8i             | %8i\n" % (confusion[0, 0], confusion[0, 1]) +
						"Notary     | %8i             | %8i\n" % (confusion[1, 1], confusion[1, 0]))

		open_file.write("\n\n")
		open_file.write("%s | %s | %s\n" %
						("Filename".rjust(45), "Reference Label".rjust(20), "Prediction Label".rjust(20)))
		for im_base_name, ref_label, pred_label in for_label_result_printing:
			open_file.write("%s | %s | %s\n" %
							(im_base_name.rjust(45), ref_label.rjust(20), pred_label.rjust(20)))


def label_to_str(label):
	return "NOTARY" if label == 1 else "NON NOTARY"


if __name__ == '__main__':
	main()
