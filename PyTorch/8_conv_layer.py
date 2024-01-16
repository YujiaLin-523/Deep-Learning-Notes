import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1,
                            padding=0)  # 彩色图像输入为3层，我们想让它的输出为6层，选3 * 3 的卷积

    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = net(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))  # 把原来6个通道拉为3个通道，为了保证所有维度总数不变，其余的分量分到第一个维度中
    writer.add_images("output", output, step)
    step = step + 1
