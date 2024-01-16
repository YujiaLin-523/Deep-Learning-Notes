import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

"""
batch_size=4 使得
img0, target0 = dataset[0]、img1, target1 = dataset[1]、
img2, target2 = dataset[2]、img3, target3 = dataset[3]，
然后这四个数据作为Dataloader的一个返回
"""

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# 用for循环取出DataLoader打包好的四个数据
writer = SummaryWriter("logs")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data  # 每个data都是由4张图片组成，imgs.size 为 [4,3,32,32]，四张32×32图片三通道，targets由四个标签组成
        writer.add_images("Epoch：{}".format(epoch), imgs, step)
        step = step + 1

writer.close()
