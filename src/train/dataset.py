"""
PhotoSense - PyTorch Dataset 数据加载器

负责:
1. 从 data/processed/{train,val,test} 加载图片
2. 应用数据增强(训练) / 标准化(验证测试)
3. 输出 (tensor[3, 224, 224], label) 给训练循环
"""

import sys
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 让 Python 能 import 我们的 categories
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dataset.categories import CATEGORY_LIST, CATEGORY_TO_IDX

DATA_DIR = Path("data/processed")

# ImageNet 标准化(用预训练模型必须用这个)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str):
    """
    根据 split 返回不同的图像变换:
    - train: 加数据增强(翻转、裁剪、颜色抖动)
    - val/test: 只做 resize + 标准化
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class PhotoSenseDataset(Dataset):
    """
    PhotoSense 图像分类数据集

    用法:
        ds = PhotoSenseDataset(split="train")
        len(ds)               # 返回总样本数
        img, label = ds[0]    # 返回 (tensor[3,224,224], int)
    """

    def __init__(self, split: str = "train", data_dir: Path = DATA_DIR):
        assert split in ("train", "val", "test"), f"split 必须是 train/val/test, 你给了 {split}"
        self.split = split
        self.transform = get_transforms(split)
        self.samples = []  # 列表: [(img_path, label_idx), ...]

        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"找不到目录 {split_dir},请先运行 split.py")

        # 扫描所有图片
        for cat_name in CATEGORY_LIST:
            cat_dir = split_dir / cat_name
            if not cat_dir.exists():
                continue
            label = CATEGORY_TO_IDX[cat_name]
            for img_path in cat_dir.glob("*.jpg"):
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"在 {split_dir} 没找到任何图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # 如果某张图损坏,返回数据集第一张作为兜底(避免训练崩)
            print(f"⚠️ 图片加载失败 {img_path}: {e},用第 0 张兜底")
            img = Image.open(self.samples[0][0]).convert("RGB")
            label = self.samples[0][1]
        img = self.transform(img)
        return img, label


def build_dataloaders(batch_size: int = 32, num_workers: int = 2):
    """构造 train/val/test 三个 DataLoader"""
    train_ds = PhotoSenseDataset("train")
    val_ds   = PhotoSenseDataset("val")
    test_ds  = PhotoSenseDataset("test")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl


# === 自检脚本(直接运行该文件时执行)===
if __name__ == "__main__":
    print("=" * 60)
    print("PhotoSense Dataset 自检")
    print("=" * 60)

    # 1. 检查三个 split 都能加载
    for split in ("train", "val", "test"):
        ds = PhotoSenseDataset(split)
        print(f"  {split:>5}: {len(ds):>6} 张图片")

    # 2. 取一张图验证
    train_ds = PhotoSenseDataset("train")
    img, label = train_ds[0]
    print()
    print(f"第一张图:")
    print(f"  tensor shape: {tuple(img.shape)} (应该是 (3, 224, 224))")
    print(f"  tensor dtype: {img.dtype}")
    print(f"  tensor min/max: {img.min():.2f} / {img.max():.2f}")
    print(f"  label: {label} ({CATEGORY_LIST[label]})")

    # 3. 验证 DataLoader 能取一个 batch
    print()
    print("测试 DataLoader (batch=8)...")
    train_dl, val_dl, test_dl = build_dataloaders(batch_size=8, num_workers=0)
    batch_imgs, batch_labels = next(iter(train_dl))
    print(f"  batch 图片 shape: {tuple(batch_imgs.shape)} (应该是 (8, 3, 224, 224))")
    print(f"  batch 标签 shape: {tuple(batch_labels.shape)} (应该是 (8,))")
    print(f"  batch 标签前 8 个: {batch_labels.tolist()}")

    print()
    print("✅ Dataset 自检通过!")