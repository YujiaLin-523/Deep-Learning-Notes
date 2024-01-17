import torch

# 对矩阵的基础操作
x = torch.arange(12)
print(x)
print(x.size())
print(x.numel())
x = torch.reshape(x, (3, 4))
print(x.size())

# 全为0和全为1的矩阵
y = torch.zeros((2, 3, 4))
z = torch.ones((2, 3, 4))
print(y)
print(z)

# 通过列表来为每个元素赋值
X = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(X)
