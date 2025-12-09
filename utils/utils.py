"""
工具函数模块
包含训练和评估所需的各种辅助函数
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def dice_coeff(pred, target, reduce_batch_first=False, epsilon=1e-6):
    """
    计算 Dice 系数（F1 分数）
    用于评估分割质量
    
    参数:
        pred: 预测结果
        target: 真实标签
        reduce_batch_first: 是否先在batch维度聚合
        epsilon: 防止除零的小数
    """
    # 计算Dice系数的平均值
    assert pred.size() == target.size()
    assert pred.dim() == 3 or not reduce_batch_first
    
    sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    
    inter = 2 * (pred * target).sum(dim=sum_dim)
    sets_sum = pred.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(pred, target, reduce_batch_first=False, epsilon=1e-6):
    """
    多类别 Dice 系数
    对每个类别分别计算Dice系数后取平均
    """
    # 对每个类别计算Dice系数
    return dice_coeff(pred.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(pred, target, multiclass=False):
    """
    Dice Loss
    1 - Dice系数
    """
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(pred, target, reduce_batch_first=True)


def evaluate(net, dataloader, device, amp=False):
    """
    在验证集上评估模型
    
    参数:
        net: 模型
        dataloader: 数据加载器
        device: 设备（CPU/GPU）
        amp: 是否使用混合精度
    
    返回:
        平均Dice系数
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    
    # 使用tqdm显示进度
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc='验证轮次', unit='batch', leave=False):
            images = batch['image']
            true_masks = batch['mask']
            
            # 移动到GPU
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            
            # 前向传播
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = net(images)
            
            # 将预测转换为概率
            if net.n_classes == 1:
                masks_pred = torch.sigmoid(masks_pred)
                masks_pred = (masks_pred > 0.5).float()
                # 计算Dice系数
                dice_score += dice_coeff(masks_pred, true_masks, reduce_batch_first=False)
            else:
                masks_pred = F.softmax(masks_pred, dim=1).argmax(dim=1)
                # 转为one-hot编码
                masks_pred = F.one_hot(masks_pred, net.n_classes).permute(0, 3, 1, 2).float()
                # 处理 true_masks 维度：去掉 channel 维度
                true_masks_squeeze = true_masks.squeeze(1).long()
                true_masks = F.one_hot(true_masks_squeeze, net.n_classes).permute(0, 3, 1, 2).float()
                # 计算多类Dice系数
                dice_score += multiclass_dice_coeff(masks_pred[:, 1:], true_masks[:, 1:], 
                                                    reduce_batch_first=False)
    
    net.train()
    return dice_score / max(num_val_batches, 1)


def save_checkpoint(state, filename='checkpoint.pth'):
    """
    保存模型检查点
    
    参数:
        state: 包含模型状态字典、优化器等信息的字典
        filename: 保存文件名
    """
    torch.save(state, filename)
    print(f'检查点已保存至 {filename}')


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    加载模型检查点
    
    参数:
        checkpoint_path: 检查点文件路径
        model: 模型
        optimizer: 优化器（可选）
    
    返回:
        起始epoch
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f'检查点已从 {checkpoint_path} 加载 (Epoch {epoch})')
    
    return epoch


class EarlyStopping:
    """
    早停机制
    当验证集性能不再提升时提前停止训练
    """
    
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        参数:
            patience: 容忍多少个epoch没有改善
            verbose: 是否打印信息
            delta: 最小改善量
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping 计数: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        """保存最佳模型"""
        if self.verbose:
            print(f'验证损失降低 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


if __name__ == '__main__':
    # 测试Dice系数计算
    pred = torch.rand(4, 1, 256, 256)
    target = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    dice = dice_coeff(pred, target)
    print(f'Dice系数: {dice:.4f}')
    
    loss = dice_loss(pred, target)
    print(f'Dice Loss: {loss:.4f}')
