"""
PhotoSense - Test Set 最终评估

加载训练好的模型,在 test set 上跑完整评估:
- Top-1 / Top-5 准确率
- 每类的准确率(找出弱点类)
- 混淆矩阵(可选)
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.train.dataset import PhotoSenseDataset
from src.train.model import create_model
from src.dataset.categories import CATEGORY_LIST

CHECKPOINT = Path("models/best_efficientnet_b3.pt")


def evaluate(model, loader, device):
    """跑一遍完整评估"""
    model.eval()

    # 总体统计
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    # 每类统计
    num_classes = len(CATEGORY_LIST)
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes

    pbar = tqdm(loader, desc="评估中", ncols=80)
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # Top-1 和 Top-5
            _, top5_pred = logits.topk(5, dim=1)
            correct_t = top5_pred.eq(labels.view(-1, 1))
            correct_top1 += correct_t[:, 0].sum().item()
            correct_top5 += correct_t.any(dim=1).sum().item()
            total += labels.size(0)

            # 每类统计
            preds = logits.argmax(dim=1)
            for label, pred in zip(labels, preds):
                per_class_total[label.item()] += 1
                if label.item() == pred.item():
                    per_class_correct[label.item()] += 1

    return {
        "top1_acc": correct_top1 / total,
        "top5_acc": correct_top5 / total,
        "total": total,
        "per_class": list(zip(CATEGORY_LIST, per_class_correct, per_class_total)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"PhotoSense Test Set 评估 | 设备: {device}")
    print("=" * 70)
    print()

    # 1. 加载模型
    print(f"[加载模型] {CHECKPOINT}")
    if not CHECKPOINT.exists():
        print(f"❌ 模型文件不存在: {CHECKPOINT}")
        sys.exit(1)

    model = create_model("efficientnet_b3", num_classes=24, pretrained=False).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  来自 epoch {ckpt['epoch']}, val_acc = {ckpt['val_acc']:.4f}")
    print()

    # 2. 加载 test set
    print("[加载 Test Set]")
    test_ds = PhotoSenseDataset("test")
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
    print(f"  测试样本: {len(test_ds)} 张")
    print()

    # 3. 评估
    print("[评估中...]")
    results = evaluate(model, test_dl, device)
    print()

    # 4. 总体结果
    print("=" * 70)
    print(f"📊 总体性能 (n={results['total']})")
    print("=" * 70)
    print(f"  Top-1 准确率: {results['top1_acc']:.4f}  ({results['top1_acc']*100:.2f}%)")
    print(f"  Top-5 准确率: {results['top5_acc']:.4f}  ({results['top5_acc']*100:.2f}%)")
    print()

    # 5. 每类性能
    print("=" * 70)
    print("📋 每类性能(按准确率排序)")
    print("=" * 70)
    per_class = sorted(
        results["per_class"],
        key=lambda x: -(x[1] / x[2] if x[2] > 0 else 0)
    )
    print(f"  {'类别':<22} {'准确率':>10}   {'正确/总数'}")
    print("  " + "-" * 50)
    for cat, correct, tot in per_class:
        acc = correct / tot if tot > 0 else 0
        bar = "█" * int(acc * 20)
        print(f"  {cat:<22} {acc*100:>8.2f}%   {correct:>3}/{tot:<3}  {bar}")

    print()
    print("=" * 70)
    print("✅ Test Set 评估完成")
    print("=" * 70)

    # 6. 保存评估报告
    report_path = Path("models/test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"PhotoSense Test Set 评估报告\n")
        f.write(f"{'='*70}\n")
        f.write(f"Top-1 准确率: {results['top1_acc']:.4f}\n")
        f.write(f"Top-5 准确率: {results['top5_acc']:.4f}\n")
        f.write(f"测试样本: {results['total']}\n")
        f.write(f"\n每类性能:\n")
        for cat, correct, tot in per_class:
            acc = correct / tot if tot > 0 else 0
            f.write(f"  {cat:<22} {acc*100:>6.2f}%   {correct}/{tot}\n")
    print(f"📄 报告已保存: {report_path}")


if __name__ == "__main__":
    main()