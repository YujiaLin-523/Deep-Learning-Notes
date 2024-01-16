import torch
import torchvision
from torch import nn
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x


net = Net()
ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = net(imgs)
        result_loss = ce_loss(outputs, targets)
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        running_loss = running_loss + result_loss
    print(running_loss)
