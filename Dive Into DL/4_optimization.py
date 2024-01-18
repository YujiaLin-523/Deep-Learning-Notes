import torch
from torch import nn


# 线性优化的简单实现
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 10)
        )


net = Net()

# 设置模型参数
net[0].weight.data.normal_(0, 0.01)  # 设置权重和正态分布的标准差，normal_是正态分布
net[0].bias.data.fill_(0)  # 初始化偏差为0

# 损失函数，使用MSELoss（均方差函数）
loss = nn.MSELoss()

# 优化器使用SGD（随机梯度下降）
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # lr是learning rate，学习率
