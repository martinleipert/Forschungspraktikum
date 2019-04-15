import torchvision.models as models
from DataSetLoader import ImageFilelist
import torch

from torch import nn
from torch import optim

TRAINING_SET = "/home/martin/Forschungspraktikum/Testdaten/Sets/equalized_set/traindata.txt"
VALIDATION_SET = "/home/martin/Forschungspraktikum/Testdaten/Sets/equalized_set/validationdata.txt"
TEST_SET = "/home/martin/Forschungspraktikum/Testdaten/Sets/equalized_set/testdata.txt"

BATCH_SIZE = 64


def main():
	# Load with self written FIle loader
	training_data = ImageFilelist('.', TRAINING_SET)
	validation_data = ImageFilelist('.', VALIDATION_SET)
	test_data = ImageFilelist('.', TEST_SET)

	# Define the DataLoader
	trainloader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
	validationloader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

	# Train on CUDA if possible
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	resnet18 = models.resnet18(pretrained=True)

	for param in resnet18.parameters():
		param.requires_grad = False

	resnet18.fc = nn.Sequential(
		nn.Linear(512, BATCH_SIZE), nn.ReLU(), nn.Dropout(0.2), nn.Linear(BATCH_SIZE, 2),  nn.LogSoftmax(dim=1))

	pass

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(resnet18.fc.parameters(), lr=0.003)
	resnet18.to(device)

	epochs = 100
	steps = 0
	running_loss = 0
	print_every = 1
	train_losses, validation_losses = [], []

	for epoch in range(epochs):
		for inputs, labels in trainloader:
			steps += 1
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()
			logps = resnet18.forward(inputs)
			loss = criterion(logps, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

			if steps % print_every == 0:
				validation_loss = 0
				accuracy = 0
				resnet18.eval()
				with torch.no_grad():
					for inputs, labels in testloader:
						inputs, labels = inputs.to(device), labels.to(device)
						logps = resnet18.forward(inputs)
						batch_loss = criterion(logps, labels)
						validation_loss += batch_loss.item()

						ps = torch.exp(logps)
						top_p, top_class = ps.topk(1, dim=1)
						equals = top_class == labels.view(*top_class.shape)
						accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
				train_losses.append(running_loss/len(trainloader))
				validation_losses.append(validation_loss/len(validationloader))

				print(f"Epoch {epoch+1}/{epochs}.. Train loss: {running_loss/print_every:.3f}.. "
				      f"Test loss: {validation_loss/len(validationloader):.3f}.. Test accuracy: {accuracy/len(validationloader):.3f}")

				running_loss = 0
				resnet18.train()
	torch.save(resnet18, 'aerialmodel.pth')
	pass


if __name__ == '__main__':
	main()
