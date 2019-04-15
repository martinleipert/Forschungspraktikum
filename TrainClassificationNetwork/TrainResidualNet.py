import torchvision.models as models
from DataSetLoader import ImageFilelist
import torch

from torch import nn
from torch import optim

TRAININGSET = "/home/martin/Forschungspraktikum/Testdaten/Sets/first_set/traindata.txt"
VALIDATIONSET = "/home/martin/Forschungspraktikum/Testdaten/Sets/first_set/validationdata.txt"
TESTSET = "/home/martin/Forschungspraktikum/Testdaten/Sets/first_set/testdata.txt"

REDUCE_DATASET = 0.1
PARTITION _EQUALLY = True

trainingdata = ImageFilelist('.', TRAININGSET)
testdata = ImageFilelist('.', TESTSET)



if REDUCE_DATASET < 1:



trainloader = torch.utils.data.DataLoader(trainingdata, batch_size=64)
testloader = torch.utils.data.DataLoader(testdata, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.requires_grad = False

resnet50.fc = nn.Sequential(nn.Linear(2048, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 10),
                        nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.003)
resnet50.to(device)

epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

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
            test_loss = 0
            accuracy = 0
            resnet50.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = resnet50.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals =  top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))

            print(f"Epoch {epoch+1}/{epochs}.. Train loss: {running_loss/print_every:.3f}.. " \
                   "Test loss: {test_loss/len(testloader):.3f}.. Test accuracy: {accuracy/len(testloader):.3f}")

            running_loss = 0
            resnet50.train()
torch.save(resnet50, 'aerialmodel.pth')
pass