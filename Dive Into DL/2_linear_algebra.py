import torch

# 通过索引访问张量中的元素
x = torch.arange(6)
print(x[2].item())
print(len(x))  # 访问张量长度

# 矩阵的转置
y = torch.arange(20).reshape(5, 4)
print(y)
print(y.T)

# 对称矩阵的转置等于自身
z = torch.tensor(([1, 2, 3], [2, 0, 4], [3, 4, 5]))
print(z.T == z)

# 复制张量
a = z.clone()
print(a == z)

# 哈达玛积 ⊙
X = torch.arange(9).reshape((3, 3))
Y = torch.arange(9).reshape((3, 3))
print(X * Y)

# 矩阵乘以常数
a = 2
x = torch.arange(24, dtype=torch.float).reshape((2, 3, 4))
print(x + a)
print((a * x).shape)

# 分维度求和
x_sum_axis0 = x.sum(dim=0)
x_sum_axis1 = x.sum(dim=1)
print(x_sum_axis0)
print(x_sum_axis1)

# 张量元素的均值
print(x.mean())
print(x.mean(axis=0))  # 按照维度求均值

# 求和时保持维度不变
print(x.sum(dim=1, keepdim=True))

# 矩阵的点积
x = torch.arange(3)
y = torch.tensor([3, 3, 3])
print(torch.dot(x, y))

# 弗洛贝尼乌斯范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

# L1范数
v = torch.arange(4, dtype=torch.float)
print(torch.abs(v).sum())
