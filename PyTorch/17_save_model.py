import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=True)
# 保存方式1
torch.save(vgg16, 'vgg16_method1.pth')

# 保存方式2(Recommended)
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')
