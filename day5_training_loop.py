import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 超参数（你调的 knobs）
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

# 2. 数据：train/val/test 三集
# 数据预处理：totensor，Normalize归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
#数据集路径
data_root = r'D:\Pythoncode\PythonProject1\datasets'
#dataset（ datasets.CIFAR10，和自己写的 MyImageDataset 一样，都继承自 Dataset）:单张图片的读取和预处理
full_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
#从训练集抽一部分做验证集
from torch.utils.data import random_split
train_size = int(0.9 * len(full_train))
val_size = len(full_train) - train_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
print(f'训练: {len(train_dataset)} | 验证: {len(val_dataset)} | 测试: {len(test_dataset)}')
#dataloader：自动分批，打乱
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 3. 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层：提取特征
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 尺寸减半

        # 全连接层：分类
        # 32×32 → 池化 → 16×16 → 池化 → 8×8
        # 32通道 × 8 × 8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #前向传播：定义数据怎么流过网络
        x = self.pool(torch.relu(self.conv1(x)))  # → [batch,16,16,16]
        x = self.pool(torch.relu(self.conv2(x)))  # → [batch,32,8,8]
        x = x.view(x.size(0), -1)                  # 展平 → [batch,2048]
        x = torch.relu(self.fc1(x))                # → [batch,128]
        x = self.fc2(x)                             # → [batch,10],不加relu
        return x                      # →  [batch, 10]。每张图 10 个分数，最高分 = 预测类别
# 创建模型实例，打印模型结构
model = SimpleCNN()
print(model)

# 4. 损失函数 + 优化器
criterion = nn.CrossEntropyLoss()  # 分类任务标配
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#  5. 训练循环
for epoch in range(EPOCHS):
    # 训练
    model.train()  #  切换到训练模式（影响 BatchNorm稳定器/Dropout随机关）
    running_loss = 0.0 # 累加这轮所有 batch 的 loss，最后除以 batch 数 = 平均 loss
    correct = 0 # 这轮猜对了多少张图
    total = 0   # 这轮总共看了多少张图
                 # correct ÷ total = 准确率

    for images, labels in train_loader:
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播，建计算图
        outputs = model(images)
        # 算 loss
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数：沿梯度反方向走一步
        optimizer.step()
        # 统计（不影响训练）
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # 取得分最高的类
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    # 验证
    model.eval()  # 评估模式
    val_correct = 0
    val_total = 0

    with torch.no_grad():  #  不计算梯度，不建计算图
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total

    print(f'Epoch [{epoch+1:2d}/{EPOCHS}] '
          f'Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')

# 6.训练完成后，用测试集评测一次
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()
test_acc = 100. * test_correct / test_total
print(f'\n最终测试准确率: {test_acc:.2f}% ')