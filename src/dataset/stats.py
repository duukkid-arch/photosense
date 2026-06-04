"""
PhotoSense - 数据集统计

显示:
- 每类有多少张
- 每张图的平均文件大小
- 总数据量(MB)
- 是否有空文件
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dataset.categories import CATEGORIES, CATEGORY_LIST

RAW_DIR = Path("data/raw")


def analyze():
    total_count = 0
    total_bytes = 0
    bad_files = []
    print(f"{'类别':<22}{'中文':<10}{'张数':>6}  {'总大小':>10}  {'平均':>8}")
    print("-" * 70)

    for cat in CATEGORY_LIST:
        cat_dir = RAW_DIR / cat
        if not cat_dir.exists():
            print(f"{cat:<22}{CATEGORIES[cat]['cn']:<10}{'---':>6}  {'未下载':>10}")
            continue

        files = list(cat_dir.glob("*.jpg"))
        count = len(files)
        cat_bytes = sum(f.stat().st_size for f in files)
        cat_mb = cat_bytes / 1024 / 1024
        avg_kb = (cat_bytes / count / 1024) if count > 0 else 0

        # 检查空文件(< 5KB,可能是损坏的)
        for f in files:
            if f.stat().st_size < 5000:
                bad_files.append(f)

        print(f"{cat:<22}{CATEGORIES[cat]['cn']:<10}{count:>6}  {cat_mb:>8.2f} MB  {avg_kb:>5.0f} KB")
        total_count += count
        total_bytes += cat_bytes

    print("-" * 70)
    print(f"{'总计':<32}{total_count:>6}  {total_bytes/1024/1024:>8.2f} MB")
    print()

    if bad_files:
        print(f"⚠️ 发现 {len(bad_files)} 个可疑文件(< 5KB):")
        for f in bad_files[:5]:
            print(f"   - {f}")
        if len(bad_files) > 5:
            print(f"   ... 还有 {len(bad_files) - 5} 个")
    else:
        print(f"✅ 没有发现可疑文件,数据质量看起来 OK")

    # 完成度提示
    target = 600
    expected_total = len(CATEGORY_LIST) * target
    progress = total_count / expected_total * 100
    print()
    print(f"📊 数据集完成度: {total_count}/{expected_total} ({progress:.1f}%)")
    if progress < 100:
        print(f"   还需下载 {expected_total - total_count} 张才能达到正式训练量")


if __name__ == "__main__":
    analyze()