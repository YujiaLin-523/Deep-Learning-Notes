import torchvision
from torch import nn

vgg16_pretrained = torchvision.models.vgg16(pretrained=True)
vgg16 = torchvision.models.vgg16(pretrained=False)
print(vgg16_pretrained)

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

vgg16_pretrained.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_pretrained)

vgg16.classifier[6] = nn.Linear(4096, 10)
print(vgg16)
