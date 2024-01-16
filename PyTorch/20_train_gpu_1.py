import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
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
if torch.cuda.is_available():
    net = net.cuda()

# 损失函数
ce_loss = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    ce_loss = ce_loss.cuda()

# 优化器
learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# 设置网络参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 设置训练的轮数
epoch = 30
# 添加tensorboard
writer = SummaryWriter('logs')

# 训练模型
net.train()
for i in range(epoch):
    print('Epoch：{}/{}'.format(i+1, epoch))
    for data in train_loader:
        # 梯度清零
        optimizer.zero_grad()
        imgs, targets = data

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = net(imgs)
        result_loss = ce_loss(outputs, targets)
        # 反向传播
        result_loss.backward()
        # 优化器调优
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("total_train_step：{}，loss：{:.4f}".format(total_train_step, result_loss.item()))
            writer.add_scalar("total_train_loss", result_loss.item(), total_train_step)

    # 测试模型
    net.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            outputs = net(imgs)
            result_loss = ce_loss(outputs, targets)
            total_test_loss = total_test_loss + result_loss.item()
            accuracy = (outputs.argmax(dim=1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("total_test_loss: {:.4f}".format(total_test_loss))
    print("test_accuracy: {:.4f}".format(accuracy / len(test_loader)))
    writer.add_scalar('total_test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', accuracy, total_test_step)
    total_test_step += 1

writer.close()
      