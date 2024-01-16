from multiprocessing import freeze_support
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sigmoid
from torch.utils.data import DataLoader

# 非线性激活tensor类型的数据
"""
input_tensor = torch.tensor([[1, -0.5],
                             [-1, 3]])
input_tensor = torch.reshape(input_tensor, (-1, 1, 2, 2))
print(input_tensor.shape)


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(x)
        return output


relu = ReLU()
output = relu(input_tensor)
print(output)
"""

# 非线性激活CiFAR-10数据集中的图片并用tensorboard展示
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)


class _ReLU(nn.Module):
    def __init__(self):
        super(_ReLU, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


relu = _ReLU()
writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("Input", imgs, step)
    output = relu(imgs)
    writer.add_images("Output", output, step)
    step = step + 1

writer.close()
