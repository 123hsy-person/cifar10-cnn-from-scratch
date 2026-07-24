import torch
import torchvision
import numpy as np

print(f"Python 版本: {__import__('sys').version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"TorchVision 版本: {torchvision.__version__}")
print(f"NumPy 版本: {np.__version__}")

# 测试 Tensor 操作
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print(f"x + y = {x + y}")

# 测试自动求导
w = torch.tensor([2.0], requires_grad=True)
z = w ** 2
z.backward()
print(f"w 的梯度: {w.grad}")  # 应该是 4.0

print("\n✅ PyTorch 环境正常！")