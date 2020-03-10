import torch
# import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.optim as optim

torch.manual_seed(1234)
ratio = 0.1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = CIFAR10(root="task2_data", train=True,
                    transform=transform, download=True)
test_set = CIFAR10(root="task2_data", train=False,
                   transform=transform, download=True)
train_set, val_set = torch.utils.data.random_split(train_set,
                                                   [int((1 - ratio) * len(train_set)),
                                                    int(ratio * len(train_set))])
test_dataset = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
train_dataset = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)


class MyModel(torch.nn.Module):
    num_features = 96 * 32 * 1
    num_categories = 10
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(MyModel.num_features, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, MyModel.num_categories)

        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()
        # self.loss = 0.0

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss_func(self, local_data, local_labels):

        output = self.forward(local_data)
        y = local_labels
        self.optimizer.zero_grad()
        self.loss = self.criterion(output, y)
        return self.loss

    def backward(self, loss):
        self.loss.backward()
        self.optimizer.step()


model = MyModel().to(device)
my_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

num_epochs = 3
for epoch in range(num_epochs):
  sum_loss = 0
  for i, (images, labels) in enumerate(train_dataset):
    images = images.to(device)
    labels = labels.to(device)

    output = model(images)
    loss = my_loss(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sum_loss += loss.item()


  print('Epoch [%d] Train Loss: %.4f'% (epoch+1, sum_loss/i))

with torch.no_grad():
  correct = total = 0
  for images, labels in test_dataset:
    images = images.to(device)
    labels = labels.to(device)

    output = model(images)
    _, predicted_labels = torch.max(output, 1)
    correct += (predicted_labels == labels).sum()
    total += labels.size(0)
  print('Percent correct: %.3f %%' % ((100 * correct) / (total + 1)))





