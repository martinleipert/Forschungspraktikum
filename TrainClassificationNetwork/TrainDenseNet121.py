import torchvision.models as models
from DataSetLoader import ImageFilelist, simple_augmentation
import torch

from torch import nn
from torch import optim

import time
import numpy

from matplotlib import pyplot

# region Parameters
CHOSEN_SET = "full_set"

TRAINING_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/traindata.txt"
VALIDATION_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/validationdata.txt"
TEST_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/testdata.txt"

LEARNING_RATE = 0.003
BATCH_SIZE = 256
NUM_CLASSES = 2

# Parametrization for the training
EPOCHS = 30
PRINT_EVERY = 25
# endregion


def main():
	t0 = time.time()

	# region Dataset Preparation
	# Load with self written FIle loader
	training_data = ImageFilelist('.', TRAINING_SET, enrich_factor=8, augmentation=simple_augmentation)
	validation_data = ImageFilelist('.', VALIDATION_SET)
	test_data = ImageFilelist('.', TEST_SET)

	# Define the DataLoader
	trainloader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
	validationloader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

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
	# Normalize
	weights = (weights / numpy.sum(weights))*len(hist)
	#endregion

	# region Model Preparation
	# Train on CUDA if possible
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Pick the model
	densenet121 = torch.load('densenet121_notarsurkunden.pth')  # models.densenet121(pretrained=True)

	for param in densenet121.parameters():
		param.requires_grad = False

	# -> Map the 1000 outputs to the 2 class problem
	densenet121.classifier = nn.Linear(1024, NUM_CLASSES)

	# Two differently parametrized loss functions for training and validation
	training_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(numpy.float32(weights)).to(device))
	validation_criterion = nn.CrossEntropyLoss().to(device)
	# Optimzer
	lr = LEARNING_RATE
	optimizer = optim.Adam(densenet121.parameters(), lr=lr)
	# To GPU (or CPU if trained on CPU)
	densenet121.to(device)
	# endregion

	# region Plotting
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
	# endregion

	# region Training
	# Running variables
	steps = 0
	running_loss = 0

	# Store losses
	train_loss, validation_losses = [], []

	t1 = time.time()

	for epoch in range(EPOCHS):

		if epoch == 15:
			lr = LEARNING_RATE * 1e-1
			optimizer = optim.Adam(densenet121.parameters(), lr=lr)

		for inputs, labels in trainloader:
			steps += 1
			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()
			logps = densenet121.forward(inputs)
			loss = training_criterion(logps, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

			if steps % PRINT_EVERY == 0:
				confusion = numpy.zeros([2, 2])

				validation_loss = 0
				accuracy = 0
				densenet121.eval()
				with torch.no_grad():
					for inputs, labels in testloader:
						inputs, labels = inputs.to(device), labels.to(device)
						logps = densenet121.forward(inputs)
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

				# Update the losses
				train_loss.append(running_loss/len(trainloader))
				validation_losses.append(validation_loss/len(validationloader))

				# Update the plots
				train_loss_curve.set_ydata(numpy.array(train_loss))
				validation_loss_curve.set_ydata(numpy.array(validation_loss))

				# Print the losses
				print(f" {t2-t0} - Epoch {epoch+1}/{EPOCHS}.. Train loss: {running_loss / PRINT_EVERY:.3f}.. "
				      f"Test loss: {validation_loss/len(validationloader):.3f}.. "
				      f"Test accuracy: {accuracy/len(validationloader):.3f}")

				# Pretty-print the confusion table
				print("\n")
				print("Doc Type   | Correctly classified | Missclassified")
				print("-" * 50)
				print("Non-notary | %8i             | %8i" % (confusion[0, 0], confusion[0, 1]))
				print("Notary     | %8i             | %8i" % (confusion[1, 1], confusion[1, 0]))

				running_loss = 0
				densenet121.train()
		# endregion

		# Save the net after each epoch
		torch.save(densenet121, 'densenet121_notarsurkunden.pth')


if __name__ == '__main__':
	main()
