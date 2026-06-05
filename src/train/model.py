"""
PhotoSense - 模型定义

提供:
- create_model: 创建 EfficientNet-B3 (ImageNet 预训练)
- count_parameters: 统计参数量
"""

import torch
import torch.nn as nn
import timm

# 默认类别数(对应 24 类摄影场景)
DEFAULT_NUM_CLASSES = 24


def create_model(
    model_name: str = "efficientnet_b3",
    num_classes: int = DEFAULT_NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    创建分类模型

    参数:
        model_name: timm 支持的模型名,如 efficientnet_b3, resnet50, vit_base_patch16_224
        num_classes: 输出类别数(我们是 24)
        pretrained: 是否加载 ImageNet 预训练权重(必须 True,不然训不动)
        dropout: 分类头 dropout(防过拟合)

    返回:
        PyTorch nn.Module 模型
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
    }


# === 自检脚本 ===
if __name__ == "__main__":
    print("=" * 60)
    print("PhotoSense Model 自检")
    print("=" * 60)

    # 1. 创建模型
    print("\n[1/4] 加载 EfficientNet-B3 (ImageNet 预训练)...")
    model = create_model("efficientnet_b3", num_classes=24, pretrained=True)
    print("✅ 模型创建成功")

    # 2. 统计参数量
    print("\n[2/4] 参数量统计")
    params = count_parameters(model)
    print(f"  总参数: {params['total_M']:.2f} M")
    print(f"  可训练: {params['trainable_M']:.2f} M")

    # 3. 前向推理测试
    print("\n[3/4] 前向推理测试")
    model.eval()  # 切到推理模式
    dummy_input = torch.randn(2, 3, 224, 224)  # batch=2 的随机图
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  输入 shape: {tuple(dummy_input.shape)} (应该是 (2, 3, 224, 224))")
    print(f"  输出 shape: {tuple(output.shape)} (应该是 (2, 24))")
    print(f"  输出 dtype: {output.dtype}")
    print(f"  输出范围: [{output.min().item():.2f}, {output.max().item():.2f}]")

    # 4. 反向传播测试(确认梯度能正确计算)
    print("\n[4/4] 反向传播测试")
    model.train()
    dummy_target = torch.tensor([5, 12])  # 假标签
    criterion = nn.CrossEntropyLoss()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    print(f"  loss = {loss.item():.4f}")
    # 检查至少有一个参数梯度不是 0
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    print(f"  梯度计算: {'✅ 正常' if has_grad else '❌ 异常'}")

    print()
    print("✅ Model 自检通过!")