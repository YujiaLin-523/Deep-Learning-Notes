import torch
import torchvision

# 加载方式1
model1 = torch.load('vgg16_method1.pth')
print(model1)

# 加载方式2
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.load_state_dict(state_dict=torch.load('vgg16_method2.pth'))
print(vgg16)
