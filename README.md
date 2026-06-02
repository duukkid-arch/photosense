# PhotoSense 📸

> 端侧细粒度图像分类 + VLM 智能修图建议系统

## 📊 项目状态

**当前阶段:Week 1 - 环境搭建与数据集**

## 🎯 计划

- [x] 项目立项 & 环境搭建
- [ ] 数据集构建(24 类,1.5 万张)
- [ ] 分类模型训练(目标 Top-1 > 90%)
- [ ] VLM 修图建议链路
- [ ] 端侧 ONNX 部署
- [ ] Web Demo 上线

## 🛠 技术栈

- **训练**: PyTorch, Timm (EfficientNet-B3)
- **VLM**: 阿里云百炼 Qwen-VL-Max
- **部署**: ONNX Runtime
- **前端**: Streamlit

## 📂 项目结构

\\\
photosense/
├── data/          # 数据集
├── models/        # 训练好的模型
├── src/           # 源代码
│   ├── dataset/   # 数据处理
│   ├── train/     # 训练脚本
│   ├── inference/ # 推理脚本
│   └── vlm/       # VLM 调用
├── notebooks/     # 实验 notebooks
└── docs/          # 文档
\\\

## 📅 开发日志

正在更新中...
