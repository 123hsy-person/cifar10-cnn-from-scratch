import torch
import numpy as np

# ===== 1. 创建 Tensor =====

# 从列表创建
a = torch.tensor([1, 2, 3, 4, 5])
print(f"从列表创建: {a}")

# 全 0
zeros = torch.zeros(3, 4)  # 3行4列
print(f"全0:\n{zeros}")

# 全 1
ones = torch.ones(2, 3)
print(f"全1:\n{ones}")

# 随机数 (0~1 均匀分布)
rand = torch.rand(2, 3)
print(f"随机:\n{rand}")

# 正态分布随机数 (均值0, 标准差1)
randn = torch.randn(2, 3)
print(f"正态随机:\n{randn}")

# 等差数列
arange = torch.arange(0, 10, 2)  # 0, 2, 4, 6, 8
print(f"等差数列: {arange}")

# 从 NumPy 转换（重要！经常用）
np_arr = np.array([1.0, 2.0, 3.0])
torch_arr = torch.from_numpy(np_arr)
print(f"从NumPy: {torch_arr}")

# ===== 2. 查看 Tensor 属性 =====

x = torch.randn(4, 3, 32, 32)  # 模拟一个 batch 的图片
print(f"\n形状: {x.shape}")      # torch.Size([4, 3, 32, 32])
print(f"维度数: {x.ndim}")       # 4
print(f"元素总数: {x.numel()}")   # 4*3*32*32 = 12288
print(f"数据类型: {x.dtype}")     # torch.float32

# ===== 3. 改变形状 =====

x = torch.randn(2, 3, 4)  # 2×3×4
print(f"\n原始: {x.shape}")

# reshape：改形状，总元素数不变
print(f"reshape (2,12): {x.reshape(2, 12).shape}")

# view：和 reshape 类似，更省内存
print(f"view (2,12): {x.view(2, 12).shape}")

# unsqueeze：增加一个维度
print(f"unsqueeze(0): {x.unsqueeze(0).shape}")  # 1×2×3×4

# squeeze：删除大小为1的维度
print(f"squeeze: {x.unsqueeze(0).squeeze(0).shape}")  # 回到 2×3×4

# ===== 4. 索引和切片（和 Python 列表一样） =====

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(f"\n原始:\n{x}")
print(f"第0行: {x[0]}")           # [1, 2, 3]
print(f"第1行第2列: {x[1, 2]}")    # 6
print(f"前两行:\n{x[:2]}")         # [[1,2,3],[4,5,6]]
print(f"第1列: {x[:, 1]}")         # [2, 5, 8]

# ===== 5. 数学运算 =====

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"\na + b = {a + b}")        # [5, 7, 9]
print(f"a * b = {a * b}")          # [4, 10, 18]  ← 逐元素乘！
print(f"a.sum() = {a.sum()}")      # 6.0
print(f"a.mean() = {a.mean()}")    # 2.0
print(f"a.max() = {a.max()}")      # 3.0

# 矩阵乘法（两种写法）
A = torch.randn(2, 3)  # 2×3
B = torch.randn(3, 4)  # 3×4
C1 = torch.matmul(A, B)  # → 2×4
C2 = A @ B                # → 2×4（更简洁）
print(f"矩阵乘结果: {C1.shape}")

# ===== 6. 设备：CPU 和 GPU =====

# 你现在只有 CPU，所以：
cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"\n设备: {cpu_tensor.device}")  # cpu

# 将来在 AutoDL 上，你需要把数据搬到 GPU：
# gpu_tensor = cpu_tensor.to('cuda')
# 或者 gpu_tensor = cpu_tensor.cuda()

# 记住铁律：模型和数据必须在同一设备上
# 现阶段都在 CPU 上，不用操心这个