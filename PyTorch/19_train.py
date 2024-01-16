import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

net = Net()
# 损失函数
ce_loss = nn.CrossEntropyLoss()
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
