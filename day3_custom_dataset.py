# 3. 自定义 Dataset 类
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
class MyImageDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir=root_dir
        self.transform=transform
        self.classes=sorted(os.listdir(root_dir))
        self.class_to_idx={name:i for i,name in enumerate(self.classes)}
        # 遍历所有图片，记录 (路径, 标签)
        self.samples = []
        # 外层 → 进每个类别文件夹
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            #中层 → 看文件夹里每个文件
            for fname in os.listdir(cls_dir):
                #内层 → 是图片就记下来（路径, 类别号）
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls_name])
                    )

        print(f"找到 {len(self.samples)} 张图片, {len(self.classes)} 个类别")
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
