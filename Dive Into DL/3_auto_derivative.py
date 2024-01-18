import torch

# 假设我们想对函数y = 2x^T * x关于x求导
x = torch.arange(4.0)
x.requires_grad_(True)  # 存储梯度，默认是None
y = 2 * torch.dot(x, x)
y.backward()  # 将梯度进行反向传播
print(x.grad)
print(x.grad == 4 * x)  # 与我们手动计算的结果进行对比

# 下面计算另一个关于x的函数的梯度
x.grad.zero_()  # PyTorch在默认情况下会累计梯度，所以要先进行梯度清零
y = x.sum()
y.backward()
print(x.grad)

# 把一些计算步骤移到计算图外
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad)
