"""
PhotoSense - 训练主循环

用法:
  python src\\train\\train.py --epochs 3 --batch_size 32        # 调试模式
  python src\\train\\train.py --epochs 20 --batch_size 64       # 正式训练(GPU)
  python src\\train\\train.py --epochs 1 --batch_size 4 --debug  # 1 batch 烟测试
"""

import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 让 Python 找到我们的模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.train.dataset import build_dataloaders
from src.train.model import create_model, count_parameters
from src.dataset.categories import CATEGORY_LIST

# === 配置 ===
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_num):
    """跑一个 epoch 的训练"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_num} [train]", ncols=100)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        # 前向 + 损失
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)

        # 反向 + 更新
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 进度条实时显示
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc": f"{correct/total:.3f}"
        })

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, split_name="val"):
    """评估模型(不算梯度)"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"          [{split_name}]", ncols=100)
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "acc": f"{correct/total:.3f}"
            })

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main(args):
    # === 1. 设备选择 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"PhotoSense 训练 | 设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)
    print()

    # === 2. 数据加载 ===
    print(f"[加载数据] batch_size={args.batch_size}, num_workers={args.num_workers}")
    train_dl, val_dl, test_dl = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  train: {len(train_dl.dataset)} 张, {len(train_dl)} batches")
    print(f"  val  : {len(val_dl.dataset)} 张, {len(val_dl)} batches")
    print()

    # === 3. 模型 ===
    print(f"[创建模型] {args.model_name}")
    model = create_model(args.model_name, num_classes=24, pretrained=True).to(device)
    params = count_parameters(model)
    print(f"  参数量: {params['total_M']:.2f} M")
    print()

    # === 4. 优化器 & 调度器 & 损失 ===
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"[训练配置]")
    print(f"  优化器: AdamW, lr={args.lr}, weight_decay=1e-4")
    print(f"  调度: CosineAnnealingLR (T_max={args.epochs})")
    print(f"  损失: CrossEntropyLoss (label_smoothing=0.1)")
    print()

    # === 5. Debug 模式: 只跑 1 个 batch ===
    if args.debug:
        print("🐛 DEBUG 模式: 只跑 1 个 batch 验证流程能跑通")
        model.train()
        imgs, labels = next(iter(train_dl))
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"  ✅ 1 个 batch 跑通 | loss = {loss.item():.4f}")
        print(f"  ✅ logits shape = {tuple(logits.shape)}")
        return

    # === 6. 正式训练循环 ===
    best_val_acc = 0.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_dl, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_dl, criterion, device, "val")

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # 打印 epoch 总结
        print()
        print(f"┌─────── Epoch {epoch}/{args.epochs} 总结 ───────┐")
        print(f"│ train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}")
        print(f"│ val_loss:   {val_loss:.4f}  val_acc:   {val_acc:.4f}")
        print(f"│ lr:         {optimizer.param_groups[0]['lr']:.6f}")
        print(f"│ epoch 用时: {epoch_time:.1f} 秒")
        print(f"└──────────────────────────────────┘")
        print()

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = MODELS_DIR / f"best_{args.model_name}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "categories": CATEGORY_LIST,
            }, save_path)
            print(f"  💾 保存最优模型: {save_path} (val_acc={val_acc:.4f})")
            print()

    # === 7. 训练完成 ===
    total_time = time.time() - t_start
    print()
    print("=" * 70)
    print(f"🎉 训练完成!")
    print(f"   总用时: {total_time/60:.1f} 分钟")
    print(f"   最优 val_acc: {best_val_acc:.4f}")
    print(f"   模型保存: {MODELS_DIR / f'best_{args.model_name}.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="efficientnet_b3")
    parser.add_argument("--debug", action="store_true", help="只跑 1 batch 验证流程")
    args = parser.parse_args()
    main(args)