import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import os

# 1. 用 torchvision 自带的 CIFAR-10 理解 Dataset
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
# 下载 CIFAR-10
data_root = r'D:\Pythoncode\PythonProject1\datasets'
train_dataset = datasets.CIFAR10(
      root=data_root,      # 存哪
      train=True,           # True=训练集(50000张)  False=测试集(10000张)
      download=True,       # 没下载就自动下，下过了就跳过
      transform=transform  # 每张图做啥处理（前面定义的）
)
test_dataset = datasets.CIFAR10(
    root=data_root,
    train=False,
    download=True,
    transform=transform
)
print(f"训练集: {len(train_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")
print(f"图片形状: {train_dataset[0][0].shape}")  # [3, 32, 32]
print(f"标签: {train_dataset[0][1]}")           # 0~9 的数字
print(f"类别名: {train_dataset.classes}")

# 2.看几眼数据长什么样
# 从 CIFAR-10 取前 10 张图，画在 2×5 的格子里，每张标上类别名

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    # 逆归一化：把 [-1,1] 变回 [0,1] 才能正常显示
    img = img / 2 + 0.5
    # [C,H,W] → [H,W,C]（matplotlib 要求的格式）
    img = img.permute(1, 2, 0)
    ax.imshow(img)
    ax.set_title(classes[label])
    ax.axis('off')
plt.tight_layout()
plt.show()

# 4. DataLoader 用法

train_loader = DataLoader(
    train_dataset,
    batch_size=64,    # 一次取 64 张
    shuffle=True,     # 训练时必须打乱！
    num_workers=0     # Windows 上设为 0，否则可能报错
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,    # 测试时不用打乱
    num_workers=0
)

# 取一个 batch 看看
images, labels = next(iter(train_loader))
print(f"\n一个 batch 的图片: {images.shape}")
# torch.Size([64, 3, 32, 32])
#            ↑   ↑  ↑   ↑
#         64张 3通道 32高 32宽
print(f"一个 batch 的标签: {labels.shape}")  # [64]
print(f"前10个标签: {labels[:10]}")