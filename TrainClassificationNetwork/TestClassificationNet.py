import torch
from torch import nn
from torchvision import models
import numpy as np
from TrainClassificationNetwork.DataSetLoader import ImageFilelist

"""
Martin Leipert
martin.leipert@fau.de

"""

SET_NAME = "equalized_set"

FILE_LIST_TEST = "/home/martin/Forschungspraktikum/Testdaten/Sets/%s/testdata.txt" % SET_NAME

MODEL_NAME = "equalized_set_resnet18_notarsurkunden_weak.pth"

BATCH_SIZE = 128
NUM_CLASSES = 2

# "RESNET_18", "RESNET_50", "DENSENET_121"
MODEL_TYPE = "RESNET_18"


def main():
	test_data = ImageFilelist(".", FILE_LIST_TEST)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, drop_last=False)

	# Evaluate on CUDA if possible
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if MODEL_TYPE == "RESNET_18":
		model_network = models.resnet18()
		model_network.fc = nn.Sequential(
			nn.Linear(512, BATCH_SIZE), nn.ReLU(), nn.Linear(BATCH_SIZE, 2), nn.LogSoftmax(dim=1))
		model_network.load_state_dict(torch.load("TrainedModels/%s" % MODEL_NAME))

	elif MODEL_TYPE == "RESNET_50":
		model_network = models.resnet50()
		model_network.fc = nn.Sequential(
			nn.Linear(2048, BATCH_SIZE), nn.ReLU(), nn.Linear(BATCH_SIZE, 2), nn.LogSoftmax(dim=1))
		model_network.load_state_dict(torch.load("TrainedModels/%s" % MODEL_NAME))

	elif MODEL_TYPE == "DENSENET_121":
		# Pick the model
		model_network = models.densenet121()
		# -> Map the 1000 outputs to the 2 class problem
		model_network.classifier = nn.Linear(1024, NUM_CLASSES)
		model_network.load_state_dict(torch.load("TrainedModels/%s" % MODEL_NAME))

	# model_network.to(device)
	torch.no_grad()

	base_name = MODEL_NAME.rstrip(".pth")

	# Loss functions for testing
	nn_loss_fct = torch.nn.NLLLoss().to(device)
	ce_loss_fct = torch.nn.CrossEntropyLoss().to(device)

	# Running variables
	loss_sum_nn = 0
	loss_sum_ce = 0

	accuracy = 0
	confusion = np.zeros((2, 2))

	# Training
	for images, labels in test_loader:
		model_network.eval()
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

		# Get the confusion matrix by a queue
		# 1,1 = Correctly classified notary documents
		for i in range(NUM_CLASSES):
			for j in range(NUM_CLASSES):
				confusion[i, j] += np.sum(np.logical_and(ref_labels == i, pred_label == j))

		# Update the losses
		loss_sum_nn += batch_loss_nn.item() / len(images)
		loss_sum_ce += batch_loss_ce.item() / len(images)

		del images, labels, logps, ps
		torch.cuda.empty_cache()

	print(f"Overall loss {loss_sum_nn}")
	denote_result(base_name, loss_sum_nn, loss_sum_ce, loss_sum_nn, confusion)


def denote_result(base_name, loss_sum, ce_loss, nn_loss, confusion):

	with open(f"Results/{base_name}_test_result.txt", "w+") as open_file:
		open_file.write(f"Loss on Test_data: {loss_sum}\n")
		open_file.write(f"Cross Entropy: {ce_loss}\n")
		open_file.write(f"NN Loss: {nn_loss}\n")
		open_file.write("\n")

		# Pretty-print the confusion table
		open_file.write("Doc Type   | Correctly classified | Missclassified\n" + "-" * 50 + "\n" +
		                "Non-notary | %8i             | %8i\n" % (confusion[0, 0], confusion[0, 1]) +
		                "Notary     | %8i             | %8i\n" % (confusion[1, 1], confusion[1, 0]))


if __name__ == '__main__':
	main()
