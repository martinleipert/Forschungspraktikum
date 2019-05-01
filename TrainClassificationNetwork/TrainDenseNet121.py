import torchvision.models as models
from DataSetLoader import ImageFilelist, simple_augmentation
import torch

from torch import nn
from torch import optim

import time
import numpy

CHOSEN_SET = "mini_set"

TRAINING_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/traindata.txt"
VALIDATION_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/validationdata.txt"
TEST_SET = f"/home/martin/Forschungspraktikum/Testdaten/Sets/{CHOSEN_SET}/testdata.txt"

BATCH_SIZE = 192
NUM_CLASSES = 2


def main():
	t0 = time.time()

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

	weights = list()
	for count in hist:
		weights.append(float(n_training_data)/float(count))

	weights = numpy.array(weights)
	# Normalize
	weights = (weights / numpy.sum(weights))*len(hist)

	# Train on CUDA if possible
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	densenet121 = models.densenet121(pretrained=True)


	for param in densenet121.parameters():
		param.requires_grad = False
	"""
	densenet121.fc = nn.Sequential(
		nn.Linear(2048, BATCH_SIZE), nn.ReLU(), nn.Dropout(0.2), nn.Linear(BATCH_SIZE, 2),  nn.LogSoftmax(dim=1))
	"""
	densenet121.classifier = nn.Linear(1024, NUM_CLASSES)
	pass

	criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(numpy.float32(weights)).to(device))
	validation_criterion = nn.CrossEntropyLoss().to(device)
	# optimizer = optim.Adam(densenet121.fc.parameters(), lr=0.003)
	optimizer = optim.Adam(densenet121.parameters(), lr=0.003)
	densenet121.to(device)

	epochs = 15
	steps = 0
	running_loss = 0
	print_every = 10
	train_losses, validation_losses = [], []

	t1 = time.time()

	for epoch in range(epochs):
		for inputs, labels in trainloader:
			steps += 1
			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()
			logps = densenet121.forward(inputs)
			loss = criterion(logps, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

			if steps % print_every == 0:
				validation_loss = 0
				accuracy = 0
				densenet121.eval()
				with torch.no_grad():
					for inputs, labels in testloader:
						inputs, labels = inputs.to(device), labels.to(device)
						logps = densenet121.forward(inputs)
						batch_loss = validation_criterion(logps, labels)
						validation_loss += batch_loss.item()

						ps = torch.exp(logps)
						top_p, top_class = ps.topk(1, dim=1)
						equals = top_class == labels.view(*top_class.shape)
						accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
				train_losses.append(running_loss/len(trainloader))
				validation_losses.append(validation_loss/len(validationloader))

				t2 = time.time()
				print(f" {t2-t0} - Epoch {epoch+1}/{epochs}.. Train loss: {running_loss/print_every:.3f}.. "
				      f"Test loss: {validation_loss/len(validationloader):.3f}.. Test accuracy: {accuracy/len(validationloader):.3f}")

				running_loss = 0
				densenet121.train()
	torch.save(densenet121, 'densenet121_notarsurkunden.pth')
	pass


if __name__ == '__main__':
	main()
