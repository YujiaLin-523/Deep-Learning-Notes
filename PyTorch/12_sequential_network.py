import torch
from torch import nn
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter


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
print(net)
test_tensor = torch.ones((64, 3, 32, 32))
output = net(test_tensor)
print(output.shape)

writer = SummaryWriter('logs')
writer.add_graph(net, test_tensor)
writer.close()
