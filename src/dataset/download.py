"""
PhotoSense - 从 Unsplash 下载分类图片

用法:
    python src\\dataset\\download.py --target 5      # 每类下载 5 张(测试)
    python src\\dataset\\download.py --target 600    # 每类下载 600 张(正式)

特性:
- 断点续传: 已下载的跳过
- 失败重试: 每张图最多 3 次
- 进度条显示
- 礼貌限速: Unsplash 免费版 50/小时
"""

import os
import sys
import time
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# 把项目根目录加入 sys.path,让我们能 import categories
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dataset.categories import CATEGORIES

# 加载 .env
load_dotenv()
ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
if not ACCESS_KEY:
    print("❌ 没有读取到 UNSPLASH_ACCESS_KEY,检查 .env")
    sys.exit(1)

# Unsplash API endpoint
UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"

# 项目数据目录
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_image_urls(query: str, page: int = 1, per_page: int = 30) -> list:
    """
    搜索 Unsplash, 返回图片 URL 列表
    返回: [{"url": "...", "id": "..."}, ...]
    """
    params = {
        "query": query,
        "page": page,
        "per_page": per_page,
        "client_id": ACCESS_KEY,
        "orientation": "landscape",  # 偏好横版(摄影常用)
    }
    try:
        resp = requests.get(UNSPLASH_SEARCH_URL, params=params, timeout=15)
        if resp.status_code == 403:
            # 限速,等一下
            print(f"\n⏳ 被限速,等 60 秒...")
            time.sleep(60)
            return fetch_image_urls(query, page, per_page)
        if resp.status_code != 200:
            print(f"\n⚠️ API 错误 {resp.status_code}: {resp.text[:200]}")
            return []
        data = resp.json()
        return [{"url": item["urls"]["regular"], "id": item["id"]}
                for item in data.get("results", [])]
    except requests.RequestException as e:
        print(f"\n⚠️ 请求失败: {e}")
        return []


def download_image(url: str, save_path: Path, max_retries: int = 3) -> bool:
    """下载单张图片,失败重试"""
    if save_path.exists():
        return True  # 断点续传:已存在的跳过

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                save_path.write_bytes(resp.content)
                return True
        except requests.RequestException:
            pass
        time.sleep(1)  # 重试前等一秒
    return False


def download_category(cat_key: str, queries: list, target: int) -> dict:
    """
    下载一个类别的图片
    返回: {"category": ..., "downloaded": N, "skipped": N, "failed": N}
    """
    save_dir = RAW_DIR / cat_key
    save_dir.mkdir(parents=True, exist_ok=True)

    # 看看已经下了多少(断点续传)
    existing = list(save_dir.glob("*.jpg"))
    already_downloaded = len(existing)

    stats = {"category": cat_key, "downloaded": 0, "skipped": already_downloaded, "failed": 0}

    if already_downloaded >= target:
        return stats  # 已经够了

    needed = target - already_downloaded
    counter = already_downloaded

    pbar = tqdm(total=needed, desc=f"  {cat_key:<22}", ncols=80)

    # 多个查询词轮流跑(增加多样性)
    for query in queries:
        if counter >= target:
            break
        for page in range(1, 20):  # 最多翻 20 页(600 张)
            if counter >= target:
                break

            urls = fetch_image_urls(query, page=page, per_page=30)
            if not urls:
                break  # 没结果或被限速

            for item in urls:
                if counter >= target:
                    break
                save_path = save_dir / f"{counter:05d}.jpg"
                if download_image(item["url"], save_path):
                    counter += 1
                    stats["downloaded"] += 1
                    pbar.update(1)
                else:
                    stats["failed"] += 1
                time.sleep(0.5)  # 礼貌限速

    pbar.close()
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5,
                        help="每个类别下载多少张(默认 5,用于测试)")
    parser.add_argument("--category", type=str, default=None,
                        help="只下载某个类别(默认全部)")
    args = parser.parse_args()

    print(f"📥 PhotoSense 数据下载")
    print(f"   目标: 每类 {args.target} 张")
    print(f"   保存到: {RAW_DIR.absolute()}")
    print()

    if args.category:
        # 只下载一个类别
        if args.category not in CATEGORIES:
            print(f"❌ 未知类别: {args.category}")
            print(f"   可选: {list(CATEGORIES.keys())}")
            sys.exit(1)
        categories_to_run = {args.category: CATEGORIES[args.category]}
    else:
        categories_to_run = CATEGORIES

    total_stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for cat_key, info in categories_to_run.items():
        stats = download_category(cat_key, info["queries"], args.target)
        total_stats["downloaded"] += stats["downloaded"]
        total_stats["skipped"]    += stats["skipped"]
        total_stats["failed"]     += stats["failed"]

    print()
    print("=" * 60)
    print(f"✅ 下载完成!")
    print(f"   新下载: {total_stats['downloaded']} 张")
    print(f"   已存在(跳过): {total_stats['skipped']} 张")
    print(f"   失败: {total_stats['failed']} 张")
    print(f"   数据目录: {RAW_DIR.absolute()}")


if __name__ == "__main__":
    main()