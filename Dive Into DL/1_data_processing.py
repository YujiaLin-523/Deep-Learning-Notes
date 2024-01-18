import torch
import os
import pandas as pd

# 对张量的基础操作
x = torch.arange(12)  # 定义一个元素为0-11的一维张量
print(x)
print(x.size())  # 输出张量的形状
print(x.numel())  # 输出张量中的元素数量
x = torch.reshape(x, (3, 4))  # 改变张量的形状
print(x.size())

# 全为0和全为1的张量
y = torch.zeros((2, 3, 4))
z = torch.ones((2, 3, 4))
print(y)
print(z)

# 通过列表来为每个元素赋值
X = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(X)
print(X.size())

# 张量的运算
print("y+z: \n", y + z)
print("y*z: \n", y * z)
Y = torch.tensor([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])
print("X中每个元素都平方得到的矩阵：\n", X ** Y)  # X的每个元素都平方

# 张量的拼接
A = torch.cat((y, z), dim=0)  # dim=0就是纵向拼接，dim=1就是横向拼接
B = torch.cat((y, z), dim=1)
print(A.size())  # 因为PyCharm输出的格式不太直观，所以这里用size展示拼接后张量的大小，可以用jupyter notebook运行这几行，看得比较清晰
print(B.size())

# 张量求和
print("A各元素的和为：\n", A.sum())  # 求和会产生一个只有一个元素的张量

# 广播机制：广播机制是把两个维度相同的张量变换成一样的形状，缺失的行或列通过复制来补齐，然后再进行运算
x = torch.arange(3).reshape(3, 1)
y = torch.arange(3).reshape(1, 3)
print(x + y)

# 切片操作
z = torch.zeros((3, 4))
z[1, 2] = 1  # 把第二行第三列的元素赋值为1
print(z)
z[0:2, :] = 1  # 把1，2行所有列赋值为1
print(z)

# 对张量进行地址不变的原地操作，可以在运算较大的张量时节省内存
X = torch.arange(12).reshape(3, 4)
Y = torch.ones((3, 4))
Z = torch.zeros_like(Y)  # 这个操作会产生一个形状与Y相同但每个元素都为0的张量
print("id(Z):", id(Z))
Z[:] = X + Y
print("id(Z):", id(Z))  # 可以发现运算前后Z的地址没有变化

# 如果后续没有用到某个张量，也可以进行自加操作来节省内存
print("id(Y):", id(Y))
Y += X
print("id(Y):", id(Y))  # 可以发现运算前后Y的地址没有变化

# 将torch张量转化为numpy张量
A = X.numpy()
print(type(A), type(X))
A = torch.tensor(A)  # 将numpy张量转化为torch张量
print(type(A), type(X))

# 将torch张量转化为python标量
a = torch.tensor([2.5])
b = a.item()  # 这个操作相当于把a中的数值拿出来
print(b)
print(float(a), int(a))  # 用强制类型转换也可以

# 数据的预处理 填补缺失值
os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 在上级路径中创建data文件夹，exist_ok参数为True表示允许文件夹已经存在
data_file = os.path.join(os.path.join('..', 'data', 'house_tiny.csv'))  # 新建一个csv文件

# 手动写入数据
with open(data_file, 'w') as f:  # w代表写入模式
    f.write("NumRooms,Alley,Price\n ")  # 列名，不能漏掉换行符，否则会都写在一行里面
    f.write("NA,Pave,127500\n")  # 这里貌似不可以在逗号后面加空格
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")

# 用pandas读取csv文件
data = pd.read_csv(data_file)
print(data)

# 插值法补全缺失的数据
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, -1]  # 隔离出输入数据中的特征值部分
inputs = inputs.fillna(inputs.mean(numeric_only=True))  # numeric_only=True的意思是只处理数值，不处理Alley这一列的字符串类
print(inputs)

# 转化Alley列的缺失值
inputs = pd.get_dummies(inputs, columns=['Alley'], dummy_na=True)
print(inputs)

# 将数据转换为torch张量
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x)
print(y)
