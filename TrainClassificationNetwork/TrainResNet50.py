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

	resnet50 = torch.load('resnet50_notarsurkunden.pth')
	# models.resnet50(pretrained=True)

	for param in resnet50.parameters():
		param.requires_grad = False

	resnet50.fc = nn.Sequential(
		nn.Linear(2048, BATCH_SIZE), nn.ReLU(), nn.Dropout(0.2), nn.Linear(BATCH_SIZE, 2),  nn.LogSoftmax(dim=1))

	pass

	criterion = nn.NLLLoss(weight=torch.from_numpy(numpy.float32(weights)).to(device))
	validation_criterion = nn.NLLLoss().to(device)
	optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.003)
	resnet50.to(device)

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
			logps = resnet50.forward(inputs)
			loss = criterion(logps, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

			if steps % print_every == 0:
				validation_loss = 0
				accuracy = 0
				resnet50.eval()
				with torch.no_grad():
					for inputs, labels in testloader:
						inputs, labels = inputs.to(device), labels.to(device)
						logps = resnet50.forward(inputs)
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
				resnet50.train()
	torch.save(resnet50, 'resnet50_notarsurkunden.pth')
	pass


if __name__ == '__main__':
	main()
