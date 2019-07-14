import torchvision.models as models
from DataSetLoader import ImageFilelist
import torch
from Augmentations.Augmentations import weak_augmentation, moderate_augmentation, heavy_augmentation
from TrainClassificationNetwork.FocalLossClassification import FocalLoss

from torch import nn
from torch import optim

import time
import numpy
import sys

from matplotlib import pyplot

import logging


# TODO implement logging -> loss and recognition
# region Parameters
CHOSEN_SET = "full_set"

TRAINING_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/traindata.txt"
VALIDATION_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/validationdata.txt"
TEST_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/testdata.txt"

LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_CLASSES = 2

# Parametrization for the training
EPOCHS = 30
PRINT_EVERY_ITERATIONS = 25
# Step size of Learning rate decay
LR_STEP_SIZE = 10
# endregion

MODEL_PATHS = {
	'RESNET_18': 'resnet18_notarsurkunden.pth',
	'RESNET_50': 'resnet50_notarsurkunden.pth',
	'DENSETNET_121': 'densenet121_notarsurkunden.pth',
}

TRAIN_FRESH = True
MODEL_NAME = "RESNET_18"
MODEL_PATH = "%s_%s" % (CHOSEN_SET, MODEL_PATHS[MODEL_NAME])

# "CE_LOSS" "NN_LOSS"
LOSS_FKT = "NN_LOSS"

"""
Setup the logger
"""
__LOG_PATH = f"TrainingLogs/Training_Log_{MODEL_NAME}_{CHOSEN_SET}.txt"
formatter = logging.Formatter('%(asctime)s\n%(message)s')

file_handler = logging.FileHandler(__LOG_PATH)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
cmd_handler = logging.StreamHandler(sys.stdout)
cmd_handler.setLevel(logging.DEBUG)
cmd_handler.setFormatter(formatter)

__LOGGER__ = logging.Logger("Training Logger")
__LOGGER__.addHandler(file_handler)
__LOGGER__.addHandler(cmd_handler)


def main():

	pyplot.ion()

	__LOGGER__.info(f"Start Training of {MODEL_NAME}\n"
					f"Training on {CHOSEN_SET}\n"
					f"Learning Rate: {LEARNING_RATE}\n"
					f"Batch size: {BATCH_SIZE}\n"
					f"Epochs: {EPOCHS}\n"
					f"Learning Rate Step Size: {LR_STEP_SIZE}\n"
					f"Loss function: {LOSS_FKT}\n")

	"""
	Prepare the data
	"""
	# Load with self written FIle loader
	training_data = ImageFilelist('.', TRAINING_SET, enrich_factor=1, augmentation=moderate_augmentation, swap=True)
	validation_data = ImageFilelist('.', VALIDATION_SET)

	# Define the DataLoader
	training_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
	validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)

	# Count the classes for weighting
	n_training_data = len(training_data.imlist)
	labels_raw = list(map(lambda x: x[1], training_data.imlist))
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

	if MODEL_NAME == "DENSETNET_121":
		# Pick the model
		model_network = models.densenet121(pretrained=TRAIN_FRESH)

		# -> Map the 1000 outputs to the 2 class problem
		model_network.classifier = nn.Linear(1024, NUM_CLASSES)

	elif MODEL_NAME == "RESNET_50":
		model_network = models.resnet50(pretrained=TRAIN_FRESH)
		model_network.fc = nn.Sequential(
			nn.Linear(2048, BATCH_SIZE), nn.ReLU(), nn.Dropout(0.2), nn.Linear(BATCH_SIZE, 2),
			nn.LogSoftmax(dim=1))

	elif MODEL_NAME == "RESNET_18":
		model_network = models.resnet18(pretrained=TRAIN_FRESH)
		model_network.fc = nn.Sequential(
			nn.Linear(512, BATCH_SIZE), nn.ReLU(), nn.Dropout(0.2), nn.Linear(BATCH_SIZE, 2),
			nn.LogSoftmax(dim=1))

	# Load state dict
	if not TRAIN_FRESH:
		model_network.load_state_dict(torch.load("TrainedModels/%s" % MODEL_PATH))

	# for param in model_network.parameters():
	#  	param.requires_grad = False

	# To GPU (or CPU if trained on CPU)
	model_network.to(device)

	# One could use the same functions for both networks?
	# Cross Entropy -> Quality of probability density function
	if LOSS_FKT is "CROSS_ENTROPY":
		# Two differently parametrized loss functions for training and validation
		training_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(numpy.float32(weights)).to(device))
		validation_criterion = nn.CrossEntropyLoss().to(device)

	# Negative Log Likelihood
	elif LOSS_FKT is "NN_LOSS":
		training_criterion = nn.NLLLoss(weight=torch.from_numpy(numpy.float32(weights)).to(device))
		validation_criterion = nn.NLLLoss().to(device)

	elif LOSS_FKT is "FOCAL_LOSS":
		training_criterion = FocalLoss().to(device)
		validation_criterion = FocalLoss().to(device)

	# Optimzer
	lr = LEARNING_RATE
	optimizer = optim.Adam(model_network.parameters(), lr=lr)
	exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.5)

	"""
	Plotting
	"""
	# Prepare the plots
	loss_fig = pyplot.figure(1, figsize=(10, 7))
	loss_ax = loss_fig.add_subplot(111)
	loss_ax.set_xlabel("Epochs")
	loss_ax.set_ylabel("")
	loss_ax.set_ylim([0, 1])
	loss_ax.set_title("Loss-Curves of %s" % MODEL_NAME)

	train_loss_curve, = loss_ax.plot([], 'b-', label="Training Loss")
	validation_loss_curve, = loss_ax.plot([], 'r-', label="Validation Loss")
	loss_fig.show()
	pyplot.pause(0.05)

	"""
	Model training
	"""
	# Running variables
	steps = 0

	# Store losses
	train_losses, validation_losses = [], []

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

	for epoch in range(EPOCHS):

		# Training
		for inputs, labels, image_paths in training_loader:
			iteration_count += 1

			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()
			logps = model_network.forward(inputs)
			loss = training_criterion(logps, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.detach().item()

			# Update the losses
			train_losses.append(running_loss / len(training_loader))
			train_loss_curve.set_xdata(range(len(train_losses)))
			train_loss_curve.set_ydata(numpy.array(train_losses))

			if iteration_count % PRINT_EVERY_ITERATIONS == 0:
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
					validation_losses.append(validation_loss / len(validation_loader))
					validation_loss_curve.set_xdata(range(len(validation_losses)))
					validation_loss_curve.set_ydata(numpy.array(validation_losses))

					loss_ax.set_xlim((0, len(validation_losses) - 1))
					pyplot.pause(0.05)
					loss_fig.savefig("TrainingLogs/%s_%s.png" % (MODEL_NAME, CHOSEN_SET), dpi=200)

					# Print the losses
					__LOGGER__.info(
						f" {t2 - t0}s total - {t2 - t1}s epoch - Epoch {epoch + 1}/{EPOCHS}.. Train loss: {running_loss / PRINT_EVERY_ITERATIONS:.3f}.. "
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
		if epoch % 5 == 4:
			torch.save(model_network.state_dict(), MODEL_PATH)

	# SAVE Model in the end
	torch.save(model_network.state_dict(), MODEL_PATH)


if __name__ == '__main__':
	main()
