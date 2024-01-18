import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets

# 设置训练的device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 首先准备一个数据集（以MNIST为例）
train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)


# 然后写一个简单的网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1568, 200),  # in_feature=32*7*7
            nn.ReLU(),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 创建网络的实例，损失函数和优化器
net = Net().to(device)
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

# 设置模型参数
epoch = 20
total_train_step = 0
total_test_step = 0
total_train_loss = 0
total_test_loss = 0

# 训练模型
for i in range(epoch):
    net.train()
    print('Epoch {}/{}'.format(i + 1, epoch))
    for data in train_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        # 定义训练时的损失函数
        train_loss = loss(outputs, labels)
        # 反向传播
        optim.zero_grad()
        train_loss.backward()
        optim.step()
        # 输出训练时的损失
        total_train_step += 1
        total_train_loss += train_loss.item()
        if total_train_step % 100 == 0:
            print('Total_train_step: {}, Loss: {:.4f}'.format(total_train_step, train_loss.item()))

# 测试模型
    net.eval()
    total_test_loss = 0
    total_correct = 0
    total_sample = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            # 定义测试时的损失函数
            test_loss = loss(outputs, labels)
            total_test_loss += test_loss.item()
            # 记录测试模型时正确的样本数，计算正确率
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_sample += labels.size(0)
            test_accuracy = total_correct / total_sample
        # 输出测试时的损失和正确率
        print("total_test_loss: {:.4f}".format(total_test_loss))
        print("test_accuracy: {:.4f}".format(test_accuracy))
        total_test_step += 1

# torch.save(net, 'model_MNIST.pth')
