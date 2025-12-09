# UNet 建筑物语义分割项目

基于 UNet 深度学习模型的航拍图像建筑物自动识别系统

## 项目概述

本项目使用 UNet 神经网络实现航拍图像中建筑物的像素级语义分割，能够自动识别和标注图像中的建筑物区域。

**核心指标**：
- 验证集 Dice Score: **0.8956** (89.56%)
- 训练轮数: 32 epochs (早停机制)
- 数据集: WHU Building Dataset

## 项目结构

```
目标识别UNet/
├── models/                 # 模型定义
│   └── unet_model.py      # UNet网络架构
├── utils/                  # 工具模块
│   ├── dataset.py         # 数据加载器
│   └── utils.py           # 评估函数和工具
├── data/                   # 数据集目录
│   ├── train/             # 训练集
│   │   ├── image/         # 训练图像
│   │   └── label/         # 训练标签
│   ├── val/               # 验证集
│   │   ├── image/         # 验证图像
│   │   └── label/         # 验证标签
│   └── test/              # 测试集
│       ├── image/         # 测试图像
│       └── label/         # 测试标签
├── checkpoints/            # 模型检查点
│   ├── best_model.pth     # 最佳模型 (Dice=0.8956)
│   └── checkpoint_epoch_*.pth
├── outputs/                # 预测输出目录
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
└── requirements.txt       # 依赖列表
```

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n unet_env python=3.8
conda activate unet_env
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖**：
- PyTorch >= 1.9.0 (CUDA 11.8)
- torchvision >= 0.10.0
- numpy >= 1.19.0
- Pillow >= 8.0.0
- matplotlib >= 3.3.0

## 使用方法

### 训练模型

```bash
python train.py --epochs 50 --batch-size 4 --learning-rate 0.0001 --scale 0.5
```

**参数说明**：
- `--epochs`: 训练轮数 (默认50)
- `--batch-size`: 批次大小 (默认4)
- `--learning-rate`: 学习率 (默认0.0001)
- `--scale`: 图像缩放比例 (默认0.5，缩放到原图一半)
- `--amp`: 启用混合精度训练
- `--classes`: 分类数量 (默认2: 背景+建筑物)

### 预测/推理

**单张图像预测**：
```bash
python predict.py -m checkpoints/best_model.pth -i data/val/image/0.tif -o outputs/ --scale 0.5
```

**批量预测**：
```bash
python predict.py -m checkpoints/best_model.pth -i data/val/image/ -o outputs/ --scale 0.5
```

**参数说明**：
- `-m, --model`: 模型权重文件路径
- `-i, --input`: 输入图像或目录
- `-o, --output`: 输出目录
- `-s, --scale`: 图像缩放 (必须与训练时一致，默认0.5)
- `-t, --threshold`: 预测阈值 (默认0.5)
- `-c, --classes`: 类别数 (默认2)

⚠️ **重要**：预测时必须使用 `--scale 0.5`，与训练时保持一致！

## 模型架构

### UNet 网络结构

```
输入 (3, 512, 512)
    ↓
编码器 (Encoder)
├── Conv1: 64 channels
├── Conv2: 128 channels
├── Conv3: 256 channels
├── Conv4: 512 channels
└── Conv5: 1024 channels
    ↓
解码器 (Decoder) + 跳跃连接
├── UpConv4: 512 channels
├── UpConv3: 256 channels
├── UpConv2: 128 channels
└── UpConv1: 64 channels
    ↓
输出 (2, 512, 512)
```

**关键特性**：
- 编码器-解码器结构
- 跳跃连接 (Skip Connection) 保留空间信息
- 双线性插值上采样
- ImageNet 标准化预处理

## 训练结果

### 性能指标

| 指标 | 数值 |
|------|------|
| 最佳 Dice Score | 0.8956 |
| 最佳 Epoch | 22 |
| 总训练轮数 | 32 (早停) |
| 训练样本数 | 4,736 张 |
| 验证样本数 | 1,036 张 |

### 损失函数

组合损失 = Cross Entropy Loss + Dice Loss

### 优化策略

- 优化器: Adam (lr=0.0001, weight_decay=1e-8)
- 学习率调度: ReduceLROnPlateau (patience=5, factor=0.5)
- 早停机制: EarlyStopping (patience=10)
- 混合精度训练: AMP (可选)
- 图像缩放: 0.5 (加速训练)

## 技术特点

1. **数据增强**: ImageNet 标准化
2. **早停机制**: 防止过拟合
3. **模型检查点**: 自动保存最佳模型
4. **GPU 加速**: CUDA 支持
5. **混合精度**: 可选 AMP 加速

## 注意事项

1. **图像缩放一致性**: 训练和预测时必须使用相同的 scale 参数
2. **归一化方式**: 使用 ImageNet 标准化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **GPU 显存**: batch_size=4 约需 8GB 显存
4. **数据格式**: 支持 .tif, .png, .jpg 格式

## 项目成果

- ✅ 完整的 UNet 语义分割系统
- ✅ 训练 Dice Score 达到 89.56%
- ✅ 支持单张/批量预测
- ✅ 完整的训练和推理流程
- ✅ 模型检查点管理

## 开发环境

- Python: 3.8
- PyTorch: 1.9+
- CUDA: 11.8
- 操作系统: Windows 11

## 参考资料

- UNet 原论文: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- 数据集: WHU Building Dataset
