import torchvision.models as models
from TrainClassificationNetwork.DataSetLoader import ImageFileList
import torch
from Augmentations.Augmentations import weak_augmentation, moderate_augmentation, heavy_augmentation
from TrainClassificationNetwork.FocalLossClassification import FocalLoss

from argparse import ArgumentParser

from torch import nn
from torch import optim

import time
import numpy
import sys
import os

from matplotlib import pyplot

import logging

# region Fixed Definitions
SET_ROOT = "/home/martin/Forschungspraktikum/Testdaten/Sets/"

# Batch size -> Given by Hardware
BATCH_SIZE = 32
# Num classes always the same for the problem
NUM_CLASSES = 2

# Fixed parameters for all models
PRINT_EVERY_ITERATIONS = 50
SAVE_EVERY_ITERATION = 200
# endregion

MODEL_PATHS = {
	'RESNET_18': 'resnet18_notarsurkunden.pth',
	'RESNET_50': 'resnet50_notarsurkunden.pth',
	'DENSETNET_121': 'densenet121_notarsurkunden.pth',
}

"""
Setup the logger
"""
formatter = logging.Formatter('%(asctime)s\n%(message)s')


cmd_handler = logging.StreamHandler(sys.stdout)
cmd_handler.setLevel(logging.DEBUG)
cmd_handler.setFormatter(formatter)

__LOGGER__ = logging.Logger("Training Logger")
__LOGGER__.addHandler(cmd_handler)


def main():
	argparser = ArgumentParser("Flexible Training Script for Notary document detection")
	argparser.add_argument("MODEL", type=str,
							help="The chosen model - available: 'RESNET_18', 'RESNET_50', 'DENSETNET_121'")
	argparser.add_argument("CHOSEN_SET", type=str, help="The directory of the defined set where the set is stored")
	argparser.add_argument("LOSS_FKT", type=str,
							help="The chosen Loss function: 'CROSS_ENTROPY', 'NN_LOSS', 'FOCAL_LOSS'")
	argparser.add_argument("AUGMENTATION", type=str, help="Selected Augmentation: 'WEAK', 'MODERATE', 'HEAVY'")
	argparser.add_argument("--trainFresh", default=False, action='store_true',
							help="Train a fresh model. If False the last state of the previous is loaded")
	argparser.add_argument("-LR", "--learningRate", type=float, default=1e-3, help="Initial Learning Rate")
	argparser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
	argparser.add_argument("--lrStep", type=int, default=1, help="Step Learning Rate after n epochs")
	argparser.add_argument("--enrichFactor", type=int, default=32,
							help="Factor to increase drawing rate of notary documents")
	argparser.add_argument("--swapSign", action='store_true', default=False, help="Augment by swapping notary sign")

	args = argparser.parse_args()

	# region Parameters
	train_fresh = args.trainFresh
	model_name = args.MODEL.upper()
	chosen_set = args.CHOSEN_SET
	learning_rate = args.learningRate
	epochs = args.epochs
	lr_step_size = args.lrStep
	loss_fkt = args.LOSS_FKT.upper()
	model_path = "%s_%s_%s" % (chosen_set, MODEL_PATHS[model_name], loss_fkt)
	augmentation_fct = args.AUGMENTATION.upper()
	drawing_factor = args.enrichFactor
	swap_sign = args.swapSign

	training_set = f"{SET_ROOT}/{chosen_set}/traindata.txt"
	validation_set = f"{SET_ROOT}/{chosen_set}/validationdata.txt"
	# endregion Parameters

	# Add the text log to the logger
	__LOG_PATH = f"TrainingLogs/Training_Log_{model_name}_{chosen_set}.txt"
	file_handler = logging.FileHandler(__LOG_PATH)
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(formatter)
	__LOGGER__.addHandler(file_handler)

	pyplot.ion()

	__LOGGER__.info(f"Start Training of {model_name}\n"
					f"Training on {chosen_set}\n"
					f"Learning Rate: {learning_rate}\n"
					f"Batch size: {BATCH_SIZE}\n"
					f"Epochs: {epochs}\n"
					f"Learning Rate Step Size: {lr_step_size}\n"
					f"Loss function: {loss_fkt}\n"
					f"Augmentation function: {augmentation_fct}\n"
					f"Swap sign: {swap_sign}\n"
					f"Drawing factor: {drawing_factor}")

	"""
	Prepare the data
	"""
	if augmentation_fct == "WEAK":
		augmentation_fct = weak_augmentation
	elif augmentation_fct == "MODERATE":
		augmentation_fct = moderate_augmentation
	elif augmentation_fct == "HEAVY":
		augmentation_fct = heavy_augmentation

	# Load with self written FIle loader
	training_data = ImageFileList('.', training_set, enrich_factor=drawing_factor, augmentation=augmentation_fct,
									swap=swap_sign)
	validation_data = ImageFileList('.', validation_set)

	# Define the DataLoader
	training_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
	validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)

	# Count the classes for weighting
	n_training_data = len(training_data.im_list)
	labels_raw = list(map(lambda x: x[1], training_data.im_list))
	hist = []
	for i in range(max(labels_raw) + 1):
		hist.append(labels_raw.count(i))

	# Calculate the weights to "balance" the dataset classes
	weights = list()
	for count in hist:
		weights.append(float(n_training_data)/float(count))

	weights = numpy.array(weights)

	# Normalize the weights
	weights = (weights / numpy.sum(weights))*len(hist)

	"""
	Model
	"""
	# Train on CUDA if possible
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if model_name == "DENSETNET_121":
		# Pick the model
		model_network = models.densenet121(pretrained=train_fresh)

		# -> Map the 1000 outputs to the 2 class problem
		model_network.classifier = nn.Linear(1024, NUM_CLASSES)

	elif model_name == "RESNET_50":
		model_network = models.resnet50(pretrained=train_fresh)
		model_network.fc = nn.Sequential(
			nn.Linear(2048, BATCH_SIZE), nn.ReLU(), nn.Dropout(0.2), nn.Linear(BATCH_SIZE, 2),
			nn.LogSoftmax(dim=1))

	elif model_name == "RESNET_18":
		model_network = models.resnet18(pretrained=train_fresh)
		model_network.fc = nn.Sequential(
			nn.Linear(512, BATCH_SIZE), nn.ReLU(), nn.Dropout(0.2), nn.Linear(BATCH_SIZE, 2),
			nn.LogSoftmax(dim=1))

	# Load state dict
	if not train_fresh:
		model_network.load_state_dict(torch.load("TrainedModels/%s" % model_path))

	# for param in model_network.parameters():
	#  	param.requires_grad = False

	# To GPU (or CPU if trained on CPU)
	model_network.to(device)

	# One could use the same functions for both networks?
	# Cross Entropy -> Quality of probability density function
	if loss_fkt == "CROSS_ENTROPY":
		# Two differently parametrized loss functions for training and validation
		training_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(numpy.float32(weights)).to(device))
		validation_criterion = nn.CrossEntropyLoss().to(device)

	# Negative Log Likelihood
	elif loss_fkt == "NLLLOSS":
		training_criterion = nn.NLLLoss(weight=torch.from_numpy(numpy.float32(weights)).to(device))
		validation_criterion = nn.NLLLoss().to(device)

	elif loss_fkt == "FOCAL_LOSS":
		training_criterion = FocalLoss().to(device)
		validation_criterion = FocalLoss().to(device)

	# Optimzer
	lr = learning_rate
	optimizer = optim.Adam(model_network.parameters(), lr=lr)
	exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.5)

	"""
	Plotting
	"""
	# Prepare the plots
	loss_fig = pyplot.figure(1, figsize=(7, 5))
	loss_ax = loss_fig.add_subplot(111)
	loss_ax.set_xlabel("Iterations")
	loss_ax.set_ylabel("Losses")
	loss_ax.set_ylim([0, 1])
	loss_ax.set_title("Loss-Curves of %s" % model_name)
	loss_ax.legend(loc=3)

	training_x_data = []
	validation_x_data = []

	train_loss_curve, = loss_ax.plot(training_x_data, [], 'b-', label="Training Loss", linewidth=2)
	validation_loss_curve, = loss_ax.plot(validation_x_data, [], 'r-', label="Validation Loss", linewidth=2)
	loss_fig.show()
	pyplot.pause(0.05)

	"""
	Model training
	"""
	# Store losses
	train_losses, validation_losses = [], []

	# Pre training validation
	validation_loss = 0
	model_network.eval()
	with torch.no_grad():
		for inputs, labels, image_paths in validation_loader:
			inputs, labels = inputs.to(device), labels.to(device)
			logps = model_network.forward(inputs)
			batch_loss = validation_criterion(logps, labels)
			validation_loss += batch_loss.item()

	validation_x_data.append(0)
	validation_losses.append(validation_loss)
	validation_loss_curve.set_xdata(validation_x_data)
	validation_loss_curve.set_ydata(numpy.array(validation_losses))

	__LOGGER__.info(f"Pre training check\n"
		f"Test loss: {validation_loss / len(validation_loader):.3f}.. ")


	t0 = time.time()
	__LOGGER__.info(
		"##########################\n"
		"#    Start Training      #\n"
		"##########################\n"
	)

	# Itzerationcounter
	iteration_count = 0
	running_loss = 0
	model_network.train()
	t1 = time.time()
	for epoch in range(epochs):

		# Training
		for inputs, labels, image_paths in training_loader:
			iteration_count += 1

			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()
			logps = model_network.forward(inputs)
			loss = training_criterion(logps, labels)
			loss.backward()
			optimizer.step()

			current_loss = loss.detach().item()
			running_loss += current_loss

			# Update the losses
			train_losses.append(current_loss)
			training_x_data.append(iteration_count)
			train_loss_curve.set_xdata(training_x_data)
			train_loss_curve.set_ydata(numpy.array(train_losses))

			if (iteration_count % PRINT_EVERY_ITERATIONS) == 0:
				del inputs, labels, image_paths

				# Validation
				confusion = numpy.zeros([2, 2])

				validation_loss = 0
				accuracy = 0
				model_network.eval()
				with torch.no_grad():
					for inputs, labels, image_paths in validation_loader:
						inputs, labels = inputs.to(device), labels.to(device)
						logps = model_network.forward(inputs)
						batch_loss = validation_criterion(logps, labels)
						validation_loss += batch_loss.item()

						# ps -> sample predictions
						ps = torch.exp(logps)
						top_p, top_class = ps.topk(1, dim=1)
						equals = top_class == labels.view(*top_class.shape)
						accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

						# Get the confusion matrix
						# Transform to CPU
						ref_labels = labels.view(*top_class.shape).cpu().detach().numpy()
						pred_label = top_class.cpu().detach().numpy()

						# Get the confusion matrix by a queue
						# 1,1 = Correctly classified notary documents
						for i in range(NUM_CLASSES):
							for j in range(NUM_CLASSES):
								confusion[i, j] += numpy.sum(numpy.logical_and(ref_labels == i, pred_label == j))

					t2 = time.time()

					# Update the plots
					validation_x_data.append(iteration_count)
					validation_losses.append(validation_loss / len(validation_loader))
					validation_loss_curve.set_xdata(validation_x_data)
					validation_loss_curve.set_ydata(numpy.array(validation_losses))

					loss_ax.set_xlim((0, iteration_count - 1))
					pyplot.pause(0.05)
					loss_fig.savefig("TrainingLogs/%s_%s.png" % (model_name, chosen_set), dpi=200)

					# Print the losses
					__LOGGER__.info(
						f" {t2 - t0}s total - {t2 - t1}s epoch - Epoch {epoch + 1}/{epochs} - "
						f"Iteration {iteration_count}\n"
						f"Train loss: {running_loss / PRINT_EVERY_ITERATIONS:.3f}.. "
						f"Test loss: {validation_loss / len(validation_loader):.3f}.. "
						f"Test accuracy: {accuracy / len(validation_loader):.3f}")

					# Pretty-print the confusion table
					__LOGGER__.info("\nDoc Type   | Correctly classified | Missclassified\n" + "-" * 50 + "\n" +
									"Non-notary | %8i             | %8i\n" % (confusion[0, 0], confusion[0, 1]) +
									"Notary     | %8i             | %8i\n" % (confusion[1, 1], confusion[1, 0]))

				t1 = time.time()
				exp_lr_scheduler.step()

				model_network.train()
				running_loss = 0

			# Save the net after each 5 epoch
			if iteration_count % SAVE_EVERY_ITERATION == (SAVE_EVERY_ITERATION - 1):
				torch.save(model_network.state_dict(), "TrainedModels/%s" % model_path)

	# SAVE Model in the end
	torch.save(model_network.state_dict(), "TrainedModels/%s" % model_path)

	with open(os.path.join(f"TrainingLogs/Loss_Curve_{model_name}_{chosen_set}.txt")) as store_file:

		store_file.write("Training Iterations:\n")
		store_file.write(training_x_data.__str__() + "\n")
		store_file.write("Training Loss:\n")
		store_file.write(train_losses.__str__() + "\n")

		store_file.write("\n")
		store_file.write("Validation Iterations:\n")
		store_file.write(validation_x_data.__str__() + "\n")
		store_file.write("Validation Loss:\n")
		store_file.write(validation_losses.__str__() + "\n")


if __name__ == '__main__':
	main()
