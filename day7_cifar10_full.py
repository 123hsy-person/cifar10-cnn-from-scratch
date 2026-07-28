import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1.超参数+数据集路径
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15
DATA_ROOT = r'D:\Pythoncode\PythonProject1\datasets'

# 2.数据：train/val/test 三集
# 训练集数据预处理（增加了数据增强）
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),        # 随机裁切：先四周垫4格0，再随机裁32×32
    transforms.RandomHorizontalFlip(),            # 随机左右翻转
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 颜色微调
    transforms.ToTensor(), #PIL图片->Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # 归一化
                         (0.2470, 0.2435, 0.2616))
])
# 测试集数据预处理（没有数据增强）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])
#从训练集抽一部分做验证集
full_train = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
test_dataset  = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_transform)
from torch.utils.data import random_split
train_size = int(0.9 * len(full_train))
val_size = len(full_train) - train_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
print(f'训练: {len(train_dataset)} | 验证: {len(val_dataset)} | 测试: {len(test_dataset)}')
#dataloader：自动分批，打乱
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# 3.定义模型，（多加了一层卷积层，以及 BatchNorm 和 Dropout）
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 每层卷积后加了 BatchNorm：bn1 = nn.BatchNorm2d(32) 跟在卷积后面，把输出标准化。让训练更稳更快
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)      # Dropout：防止过拟合

        # 3 次池化后：32→16→8→4
        # 128通道 × 4 × 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # 卷积->标准化->激活->池化,x=[batch,32,16,16]
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # [batch,64,8,8]
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # [batch,128,4,4]
        x = x.view(x.size(0), -1) #展平-> [batch,2048]
        x = self.dropout(torch.relu(self.fc1(x))) # Dropout 加在全连接层前面，训练时随机关 30% 神经元，防止过拟合。
        x = self.fc2(x)
        return x
# 创建模型实例，打印模型结构
device = torch.device('cpu')  # 设备是 CPU，autodl上改成（'cuda'）
model = ImprovedCNN().to(device) #模型在cpu上
print(f'总参数量: {sum(p.numel() for p in model.parameters()):,}')

# 4.损失函数 + 优化器 + 学习率调度
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# StepLR：每 7 个 epoch 把学习率 × 0.1（先大步快走，再小步微调）

# 5. 训练循环
# 存每个 epoch 的结果，训练完画曲线用
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0         # 记录最高验证准确率
patience_counter = 0       # 早停计数器
EARLY_STOP_PATIENCE = 5    # 连续 5 个 epoch 不提升就停

for epoch in range(EPOCHS):
    # 训练模式
    model.train()
    train_loss=0.0 # 累加这轮所有 batch 的 loss，最后除以 batch 数 = 平均 loss
    correct=0  # 这轮猜对了多少张图
    total=0 # 这轮总共看了多少张图
    for images, labels in train_loader:
        #  模型和数据必须在同一设备
        images, labels = images.to(device), labels.to(device)
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播，建计算图
        outputs = model(images)
        # 算loss
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        train_loss += loss.item() # loss是tensor，item()转数字
        #   outputs.shape = [64, 10]
        #   outputs.max(1)  → 沿 dim=1（10 个类别方向）
        _, predicted = outputs.max(1) # .max()默认返回值和位置，不要值
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_losses.append(train_loss / len(train_loader))
    train_accs.append(100. * correct / total)
    # 验证
    model.eval() # 评估模式
    val_loss=0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(100. * val_correct / val_total)
