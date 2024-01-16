import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 继承父类的初始化

    def forward(self, input):
        output = input + 1
        return output


net = Net()
x = torch.tensor(1.0)  # 创建一个值为1.0的tensor变量
output = net(x)
print(output)
