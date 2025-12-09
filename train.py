"""
UNet 模型训练脚本
用于训练建筑物分割模型
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# 导入自定义模块
from models.unet_model import UNet
from utils.dataset import BuildingDataset, get_transform
from utils.utils import evaluate, save_checkpoint, dice_loss, EarlyStopping


def train_model(
    model,
    device,
    train_loader,
    val_loader,
    epochs=50,
    batch_size=4,
    learning_rate=1e-4,
    save_checkpoint_path='./checkpoints/',
    amp=False
):
    """
    训练模型
    
    参数:
        model: UNet模型
        device: 训练设备
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_checkpoint_path: 检查点保存路径
        amp: 是否使用混合精度训练
    """
    # 创建保存目录
    Path(save_checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    # 定义损失函数
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # 混合精度训练
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # 早停
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # 全局步数
    global_step = 0
    best_dice = 0
    
    print(f'''开始训练:
        训练轮数:     {epochs}
        批次大小:     {batch_size}
        学习率:       {learning_rate}
        训练样本数:   {len(train_loader.dataset)}
        验证样本数:   {len(val_loader.dataset)}
        检查点保存:   {save_checkpoint_path}
        设备:         {device.type}
        混合精度:     {amp}
    ''')
    
    # 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                
                # 前向传播
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    
                    if model.n_classes == 1:
                        # 二分类
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            torch.sigmoid(masks_pred),
                            true_masks,
                            multiclass=False
                        )
                    else:
                        # 多分类
                        loss = criterion(masks_pred, true_masks.squeeze(1).long())
                        loss += dice_loss(
                            torch.softmax(masks_pred, dim=1).float(),
                            torch.nn.functional.one_hot(true_masks.squeeze(1).long(), model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                
                # 反向传播
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                # 更新进度条
                pbar.update(1)
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
        
        # 验证
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)
        
        print(f'Epoch {epoch} - 训练损失: {epoch_loss/len(train_loader):.4f}, 验证Dice: {val_score:.4f}')
        
        # 保存最佳模型
        if val_score > best_dice:
            best_dice = val_score
            checkpoint_path = os.path.join(save_checkpoint_path, 'best_model.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice_score': val_score,
            }, checkpoint_path)
            print(f'最佳模型已保存！Dice Score: {val_score:.4f}')
        
        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(save_checkpoint_path, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice_score': val_score,
            }, checkpoint_path)
        
        # 早停检查
        early_stopping(1 - val_score, model, os.path.join(save_checkpoint_path, 'early_stop.pth'))
        if early_stopping.early_stop:
            print("早停触发，停止训练")
            break
    
    print(f'训练完成！最佳 Dice Score: {best_dice:.4f}')


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练 UNet 建筑物分割模型')
    
    parser.add_argument('--epochs', '-e', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='批次大小')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help='学习率')
    parser.add_argument('--load', '-f', type=str, default=False, help='从检查点加载模型')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='图像缩放比例')
    parser.add_argument('--amp', action='store_true', default=False, help='使用混合精度训练')
    parser.add_argument('--bilinear', action='store_true', default=False, help='使用双线性插值上采样')
    parser.add_argument('--classes', '-c', type=int, default=2, help='分类数量')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(device=device)
    
    print(f'网络参数:\n'
          f'\t输入通道数: 3\n'
          f'\t输出类别数: {args.classes}\n'
          f'\t双线性上采样: {args.bilinear}')
    
    # 加载预训练模型
    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'模型已从 {args.load} 加载')
    
    # 准备数据集
    train_dataset = BuildingDataset(
        image_dir='./data/train/image',
        mask_dir='./data/train/label',
        scale=args.scale,
        transform=get_transform(train=True)
    )
    
    val_dataset = BuildingDataset(
        image_dir='./data/val/image',
        mask_dir='./data/val/label',
        scale=args.scale,
        transform=get_transform(train=False)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows下建议设为0
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    # 开始训练
    try:
        train_model(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            amp=args.amp
        )
    except KeyboardInterrupt:
        print('训练被用户中断')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
