from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets
import torch

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, x):
        output = self.linear1(x)
        return output


net = Net()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = net(output)
    print(output.shape)
