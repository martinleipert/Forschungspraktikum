import torchvision.models as models
from DataSetLoader import ImageFilelist
import torch
from Augmentations.Augmentations import weak_augmentation

from torch import nn
from torch import optim

import time
import numpy

from matplotlib import pyplot

# region Parameters
CHOSEN_SET = "mini_set"

TRAINING_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/traindata.txt"
VALIDATION_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/validationdata.txt"
TEST_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/testdata.txt"

LEARNING_RATE = 5e-4
BATCH_SIZE = 128
NUM_CLASSES = 2

# Parametrization for the training
EPOCHS = 60
PRINT_EVERY = 25
# endregion

MODEL_PATHS = {
	'RESNET_18': 'resnet18_notarsurkunden.pth',
	'RESNET_50': 'resnet50_notarsurkunden.pth',
	'DENSETNET_121': 'densenet121_notarsurkunden.pth',
}

TRAIN_FRESH = True
MODEL_NAME = "RESNET_18"
MODEL_PATH = "%s_%s" % (CHOSEN_SET, MODEL_PATHS[MODEL_NAME])


def main():
	t0 = time.time()

	"""
	Prepare the data
	"""
	# Load with self written FIle loader
	training_data = ImageFilelist('.', TRAINING_SET, enrich_factor=8, augmentation=weak_augmentation)
	validation_data = ImageFilelist('.', VALIDATION_SET)
	test_data = ImageFilelist('.', TEST_SET)

	# Define the DataLoader
	training_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
	validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

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

	if TRAIN_FRESH is True:
		if MODEL_NAME == "DENSETNET_121":
			# Pick the model
			model_network = models.densenet121(pretrained=True)
			# -> Map the 1000 outputs to the 2 class problem
			model_network.classifier = nn.Linear(1024, NUM_CLASSES)

		elif MODEL_NAME == "RESNET_50":
			model_network = models.resnet50(pretrained=True)
			model_network.fc = nn.Sequential(
				nn.Linear(2048, BATCH_SIZE), nn.ReLU(), nn.Dropout(0.2), nn.Linear(BATCH_SIZE, 2),
				nn.LogSoftmax(dim=1))

		elif MODEL_NAME == "RESNET_18":
			model_network = models.resnet18(pretrained=True)
			model_network.fc = nn.Sequential(
				nn.Linear(512, BATCH_SIZE), nn.ReLU(), nn.Dropout(0.2), nn.Linear(BATCH_SIZE, 2),
				nn.LogSoftmax(dim=1))

		# for param in model_network.parameters():
		#  	param.requires_grad = False
	else:
		model_network = torch.load(MODEL_PATH)

	# To GPU (or CPU if trained on CPU)
	model_network.to(device)

	if MODEL_NAME in ["DENSENET_121"]:
		# Two differently parametrized loss functions for training and validation
		training_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(numpy.float32(weights)).to(device))
		validation_criterion = nn.CrossEntropyLoss().to(device)

	elif MODEL_NAME in ["RESNET_18", "RESNET_50"]:
		training_criterion = nn.NLLLoss(weight=torch.from_numpy(numpy.float32(weights)).to(device))
		validation_criterion = nn.NLLLoss().to(device)


	# Optimzer
	lr = LEARNING_RATE
	optimizer = optim.Adam(model_network.parameters(), lr=lr)

	"""
	Plotting
	"""
	# Prepare the plots
	loss_fig = pyplot.figure(1, figsize=(10, 7))
	loss_ax = loss_fig.add_subplot(111)
	loss_ax.set_xlabel("Epochs")
	loss_ax.set_ylabel("")
	loss_ax.set_ylim([0, 1])
	loss_ax.set_title("Loss-Curves of DenseNet121")

	train_loss_curve, = loss_ax.plot([], 'b-', label="Training Loss")
	validation_loss_curve, = loss_ax.plot([], 'r-', label="Validation Loss")
	loss_fig.show()

	"""
	Model training
	"""
	# Running variables
	steps = 0
	running_loss = 0

	# Store losses
	train_loss, validation_losses = [], []

	t1 = time.time()

	print(
		"##########################\n"
		"#    Start Training      #\n"
		"##########################\n"
	)

	for epoch in range(EPOCHS):

		if epoch == 15:
			lr = LEARNING_RATE * 1e-1
			optimizer = optim.Adam(model_network.parameters(), lr=lr)

		model_network.train()
		running_loss = 0
		# Training
		for inputs, labels in training_loader:
			steps += 1
			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()
			logps = model_network.forward(inputs)
			loss = training_criterion(logps, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

			# Update the losses
			train_loss.append(running_loss / len(training_loader))
			train_loss_curve.set_ydata(numpy.array(train_loss))


		# Validation
		confusion = numpy.zeros([2, 2])

		validation_loss = 0
		accuracy = 0
		model_network.eval()
		with torch.no_grad():
			for inputs, labels in validation_loader:
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
				ref_labels = numpy.array(labels.view(*top_class.shape).cpu())
				pred_label = numpy.array(top_class.cpu())

				# Get the confusion matrix by a queue
				# 1,1 = Correctly classified notary documents
				for i in range(NUM_CLASSES):
					for j in range(NUM_CLASSES):
						confusion[i, j] += numpy.sum(numpy.logical_and(ref_labels == i, pred_label == j))

			t2 = time.time()

			validation_losses.append(validation_loss/len(validation_loader))

			# Update the plots
			validation_loss_curve.set_ydata(numpy.array(validation_loss))

			# Print the losses
			print(f" {t2-t0} - Epoch {epoch+1}/{EPOCHS}.. Train loss: {running_loss / PRINT_EVERY:.3f}.. "
				f"Test loss: {validation_loss/len(validation_loader):.3f}.. "
				f"Test accuracy: {accuracy/len(validation_loader):.3f}")

			# Pretty-print the confusion table
			print("\n")
			print("Doc Type   | Correctly classified | Missclassified")
			print("-" * 50)
			print("Non-notary | %8i             | %8i" % (confusion[0, 0], confusion[0, 1]))
			print("Notary     | %8i             | %8i" % (confusion[1, 1], confusion[1, 0]))

		# endregion

		# Save the net after each 5 epoch
		if epoch % 5 == 4:
			torch.save(model_network, MODEL_PATH)

	# SAVE Model in the end
	torch.save(model_network, MODEL_PATH)


if __name__ == '__main__':
	main()
