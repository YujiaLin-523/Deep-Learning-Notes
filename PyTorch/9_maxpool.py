from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# 使用最大池化处理tensor数据
"""
input_tensor = torch.tensor([[1, 2, 0, 3, 1],
                             [0, 1, 2, 3, 1],
                             [1, 2, 1, 0, 0],
                             [5, 2, 3, 1, 1],
                             [2, 1, 0, 1, 1]], dtype=torch.float32)

input_tensor = torch.reshape(input_tensor, (-1, 1, 5, 5))
print(input_tensor.shape)


class Maxpool2d(nn.Module):
    def __init__(self):
        super(Maxpool2d, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool(x)
        return output


maxpool = Maxpool2d()
output = maxpool(input_tensor)
print(output)
"""

# 使用最大池化处理图像数据
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)


class Maxpool2d(nn.Module):
    def __init__(self):
        super(Maxpool2d, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool(x)
        return output


maxpool = Maxpool2d()
writer = SummaryWriter('logs')
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_image('input', torchvision.utils.make_grid(imgs), step)
    output = maxpool(imgs)
    writer.add_image('output', torchvision.utils.make_grid(output), step)
    step += 1
