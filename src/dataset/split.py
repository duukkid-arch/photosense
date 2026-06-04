"""
PhotoSense - 数据集划分

把 data/raw/ 按 8:1:1 划分到 data/processed/train|val|test/
"""

import sys
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dataset.categories import CATEGORY_LIST

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


def split_data(train_ratio=0.8, val_ratio=0.1, seed=42, force=False):
    """
    划分数据集
    train_ratio + val_ratio + test_ratio 必须等于 1
    """
    test_ratio = 1 - train_ratio - val_ratio
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    print(f"📊 数据集划分")
    print(f"   train: {train_ratio:.0%} | val: {val_ratio:.0%} | test: {test_ratio:.0%}")
    print(f"   随机种子: {seed}")
    print()

    # 如果 processed 已存在,询问是否覆盖
    if OUT_DIR.exists():
        if not force:
            print(f"⚠️ {OUT_DIR} 已存在")
            ans = input("   要清空并重新划分吗? (y/N): ").strip().lower()
            if ans != "y":
                print("取消划分")
                return
        shutil.rmtree(OUT_DIR)

    random.seed(seed)
    stats = {"train": 0, "val": 0, "test": 0}

    for cat in CATEGORY_LIST:
        src_dir = RAW_DIR / cat
        if not src_dir.exists():
            print(f"⚠️ 跳过 {cat}(未下载)")
            continue

        files = sorted(src_dir.glob("*.jpg"))
        random.shuffle(files)
        n = len(files)

        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        # test 取剩下的(避免凑整误差导致漏掉)

        splits = {
            "train": files[:n_train],
            "val":   files[n_train : n_train + n_val],
            "test":  files[n_train + n_val :],
        }

        for split_name, split_files in splits.items():
            target_dir = OUT_DIR / split_name / cat
            target_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, target_dir / f.name)
            stats[split_name] += len(split_files)

        print(f"  {cat:<22} {n:>4} → train {len(splits['train']):>3}  val {len(splits['val']):>2}  test {len(splits['test']):>2}")

    print()
    print("=" * 60)
    print(f"✅ 划分完成!")
    print(f"   train: {stats['train']} 张")
    print(f"   val:   {stats['val']} 张")
    print(f"   test:  {stats['test']} 张")
    print(f"   总计:  {sum(stats.values())} 张")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--force", action="store_true", help="不询问,直接覆盖")
    args = parser.parse_args()
    split_data(seed=args.seed, force=args.force)