"""
PhotoSense - 24 类摄影场景分类定义

每个类别包含:
- key: 内部代号(英文,目录名用)
- cn: 中文显示名
- queries: Unsplash 搜索关键词(用于下载)
"""

CATEGORIES = {
    # 人像
    "portrait_indoor":     {"cn": "室内人像", "queries": ["indoor portrait", "studio portrait"]},
    "portrait_outdoor":    {"cn": "户外人像", "queries": ["outdoor portrait", "park portrait"]},
    "portrait_night":      {"cn": "夜景人像", "queries": ["night portrait", "low light portrait"]},
    "selfie":              {"cn": "自拍",     "queries": ["selfie", "mirror selfie"]},

    # 风光
    "landscape_mountain":  {"cn": "山景",     "queries": ["mountain landscape", "alpine view"]},
    "landscape_sea":       {"cn": "海景",     "queries": ["sea ocean", "beach coastline"]},
    "landscape_sky":       {"cn": "天空云",   "queries": ["sky clouds", "dramatic sky"]},
    "sunset_sunrise":      {"cn": "日落日出", "queries": ["sunset", "sunrise"]},

    # 城市
    "city_skyline":        {"cn": "城市天际线", "queries": ["city skyline", "urban view"]},
    "architecture":        {"cn": "建筑",     "queries": ["architecture", "modern building"]},
    "street":              {"cn": "街拍",     "queries": ["street photography", "urban street"]},
    "night_city":          {"cn": "夜景城市", "queries": ["city night lights", "neon street"]},

    # 美食
    "food_chinese":        {"cn": "中餐",     "queries": ["chinese food", "asian cuisine"]},
    "food_western":        {"cn": "西餐",     "queries": ["western food", "plated dish"]},
    "dessert":             {"cn": "甜品",     "queries": ["dessert cake", "pastry"]},
    "drink":               {"cn": "饮品",     "queries": ["coffee cup", "cocktail drink"]},

    # 生物
    "pet_cat":             {"cn": "猫",       "queries": ["cat closeup", "kitten"]},
    "pet_dog":             {"cn": "狗",       "queries": ["dog closeup", "puppy"]},
    "flower":              {"cn": "花卉",     "queries": ["flower closeup", "blossom"]},

    # 室内/产品
    "indoor_home":         {"cn": "室内家居", "queries": ["interior home", "living room"]},
    "product":             {"cn": "产品静物", "queries": ["product photography", "still life"]},

    # 特殊场景
    "sports":              {"cn": "运动",     "queries": ["sports action", "athlete"]},
    "concert":             {"cn": "演唱会",   "queries": ["concert stage", "music festival"]},
    "rain":                {"cn": "雨天",     "queries": ["rainy day", "raindrops"]},
}

# 类别数验证
assert len(CATEGORIES) == 24, f"应该是 24 类,实际 {len(CATEGORIES)} 类"

# 给类别一个固定的索引(训练时用)
CATEGORY_LIST = sorted(CATEGORIES.keys())
CATEGORY_TO_IDX = {name: idx for idx, name in enumerate(CATEGORY_LIST)}
IDX_TO_CATEGORY = {idx: name for idx, name in enumerate(CATEGORY_LIST)}


if __name__ == "__main__":
    print(f"总类别数: {len(CATEGORIES)}")
    print()
    print(f"{'索引':<5}{'英文 key':<22}{'中文':<10}{'查询词数'}")
    print("-" * 55)
    for idx, key in enumerate(CATEGORY_LIST):
        info = CATEGORIES[key]
        print(f"{idx:<5}{key:<22}{info['cn']:<10}{len(info['queries'])}")