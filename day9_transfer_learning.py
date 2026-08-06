import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# 1.超参数+数据集路径
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DATA_ROOT = r'D:\Pythoncode\PythonProject1\datasets'
device = torch.device('cpu')

# 2.数据：train/val/test 三集
# 训练集数据预处理(数据增强+ResNet 需要 224×224 输入)
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),            # 训练时加一点增强,随机左右翻转
    transforms.ToTensor(),                        #PIL图片->Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# 测试集数据预处理（没有数据增强）
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# 创建两个 Dataset，指向同一批图片
# 训练集：带数据增强
full_train_aug = datasets.CIFAR10(root=DATA_ROOT, train=True,download=True, transform=train_transform)
# 验证集用：不带数据增强（重要！）
full_train_eval = datasets.CIFAR10(root=DATA_ROOT, train=True,download=True, transform=test_transform)
# 测试集：不带数据增强
test_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False,download=True, transform=test_transform)

# 用同一组索引切分，保证训练/验证互不重叠
from torch.utils.data import Subset
indices = list(range(len(full_train_aug)))
train_size = int(0.9 * len(indices))
val_size = len(indices) - train_size
train_dataset = Subset(full_train_aug, indices[:train_size])     # 带增强
val_dataset   = Subset(full_train_eval, indices[train_size:])    # 不带增强

print(f'训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}')
#dataloader：自动分批，打乱
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 3.加载预训练 ResNet-18和替换最后一层
model = models.resnet18(weights='IMAGENET1K_V1')

# 冻结所有卷积层参数（不让它们训练）
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层：ImageNet 1000 类 → CIFAR-10 10 类
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# 数一下可训练参数
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'可训练参数: {trainable:,} / 总参数: {total:,} ({100*trainable/total:.1f}%)')

# 4.损失函数 + 优化器 + 学习率调度
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 5. 训练循环

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

        train_loss += loss.item()  # loss是tensor，item()转数字
        #   outputs.shape = [64, 10]
        #   outputs.max(1)  → 沿 dim=1（10 个类别方向）
        _, predicted = outputs.max(1)  # .max()默认返回值和位置，不要值
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total

    # 验证
    model.eval()
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

    val_acc = 100. * val_correct / val_total

    # 保存验证准确率最高的模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_transfer_model.pth')
        patience_counter = 0
        print(f' 保存最优模型: {best_val_acc:.2f}%')
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f' 早停：连续 {EARLY_STOP_PATIENCE} 个 epoch 验证准确率没提升')
            break

    scheduler.step()

    print(f'Epoch [{epoch+1:2d}/{EPOCHS}] '
          f'Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | '
          f'Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%')

# 6.训练完成后，加载最优模型，用测试集评测一次
model.load_state_dict(torch.load('best_transfer_model.pth'))
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = 100. * test_correct / test_total
print(f'\n最终测试准确率: {test_acc:.2f}% ')
print(f'最佳验证准确率: {best_val_acc:.2f}%')

print('\n 迁移学习完成！')
print('从零训 CNN ≈ 70-75%，迁移学习 ResNet-18 ≈ 85-90%')
print('数据更少、训练更快、效果更好 → 这就是迁移学习的威力')
