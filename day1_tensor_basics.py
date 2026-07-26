import torch
import numpy as np

# 创建tensor

# 从列表创建
a = torch.tensor([1,2,3,4,5])
print(f"从列表创建a =\n {a}")

# 全零
zeros=torch.zeros(3,4)
print(f"全零zeros = \n{zeros}")

# 全1
ones=torch.ones(2)
print(f"全1ones = \n{ones}")

# 随机数（0-1均匀分布）
rand=torch.rand(2)
print(f"随机rand = \n{rand}")

# 正态分布随机数（均值0，标准差1）
randn=torch.randn(2,3)
print(f"正态随机randn\n{randn}")

# 等差数列
arange=torch.arange(0,10,2)
print(f"等差数列arange=\n{arange}")

# 从numpy转换
np_arr=np.array([1,2,3,4])
torch_arr=torch.from_numpy(np_arr)
print(f"numpy数组转为tensor\n{torch_arr}")

# 查看 Tensor 属性
x=torch.randn(4,3,32,32)
print(f"形状：{x.shape}")
print(f"维度数：{x.ndim}")
print(f"元素总数{x.numel()}")
print(f"数据类型{x.dtype}")

# 改变形状
x=torch.randn(2,3,4)
print(f"未改变前形状{x.shape}")
print(f"用reshape(2,12)改变后的形状{x.reshape(2,12).shape}")
print(f"用view(4,6)改变后的形状{x.view(4,6).shape}")
print(f"在(0)位置增加1个维度:{x.unsqueeze(0).shape}")
print(f"在(0)位置删除1个维度:{x.unsqueeze(0).squeeze(0).shape}")

# 索引和切片
x=torch.tensor([[1,2,3],
               [4,5,6],
               [7,8,9]])
print(f"打印整个x：\n{x}")
print(f"第0行{x[0]}")
print(f"第0列{x[:,0]}")
print(f"第1行第2列{x[1,2]}")
print(f"前两行{x[:2]}")

# 数学运算
a=torch.tensor([1.0,2.0,3.0])
b=torch.tensor([4.0,5.0,6.0])
print(f"a+b = {a+b}")
print(f"a*b = {a*b}")
print(f"a.sum={a.sum()}")
print(f"a.mean={a.mean()}")
print(f"a.max={a.max()}")

# 矩阵乘法
A=torch.tensor([[1.0,2.0],[3.0,4.0]]) #2行2列
B=torch.tensor([[5.0,6.0,9.0],[7.0,8.0,10.0]]) #2行3列
C1=torch.matmul(A,B)
C2=A@B
print(f"C1=\n{C1}")
print(f"C2=\n{C2}")

# 设备
cpu_tesor=torch.tensor([1.0,2.0])
print(f"设备：{cpu_tesor.device}")

# 自动求导
x=torch.tensor([1.0,2.0],requires_grad=True)
y=x[0]**2+x[1]**3
print(f"y={y}")
y.backward()
print(f"x.grad={x.grad}")