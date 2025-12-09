"""
数据集加载模块
用于加载建筑物分割数据集
"""
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BuildingDataset(Dataset):
    """
    建筑物分割数据集类
    
    参数:
        image_dir: 图像文件夹路径
        mask_dir: 掩码文件夹路径
        scale: 图像缩放比例 (0-1之间)
        transform: 图像变换
    """
    
    def __init__(self, image_dir, mask_dir, scale=1.0, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.scale = scale
        self.transform = transform
        
        # 获取所有图像文件名
        self.images = [f for f in os.listdir(image_dir) 
                       if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        if len(self.images) == 0:
            raise RuntimeError(f'在 {image_dir} 中未找到任何图像文件')
        
        print(f'创建数据集，共 {len(self.images)} 张图像')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 加载图像
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 尝试找到对应的mask文件
        mask_name = img_name
        # 可能需要根据实际数据集调整mask文件名
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            potential_mask = os.path.splitext(img_name)[0] + ext
            if os.path.exists(os.path.join(self.mask_dir, potential_mask)):
                mask_name = potential_mask
                break
        
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 读取图像和掩码
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转为灰度图
        
        # 缩放图像
        if self.scale != 1.0:
            new_width = int(image.width * self.scale)
            new_height = int(image.height * self.scale)
            image = image.resize((new_width, new_height), Image.BILINEAR)
            mask = mask.resize((new_width, new_height), Image.NEAREST)
        
        # 转换为numpy数组
        image = np.array(image)
        mask = np.array(mask)
        
        # 二值化mask: 将非零值设为1，背景设为0
        mask = (mask > 0).astype(np.float32)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            # 默认变换：转为tensor并归一化
            image = transforms.ToTensor()(image)
        
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'filename': img_name
        }


def get_transform(train=True):
    """
    获取数据增强变换
    
    参数:
        train: 是否为训练集
    """
    if train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    return transform


if __name__ == '__main__':
    # 测试数据集
    import matplotlib.pyplot as plt
    
    # 示例路径，需要根据实际情况修改
    image_dir = '../data/train/image'
    mask_dir = '../data/train/label'
    
    if os.path.exists(image_dir) and os.path.exists(mask_dir):
        dataset = BuildingDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            scale=1.0
        )
        
        # 查看第一个样本
        sample = dataset[0]
        print(f"图像形状: {sample['image'].shape}")
        print(f"掩码形状: {sample['mask'].shape}")
        print(f"文件名: {sample['filename']}")
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(sample['image'].permute(1, 2, 0))
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        axes[1].imshow(sample['mask'].squeeze(), cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_sample.png')
        print("示例图像已保存为 dataset_sample.png")
    else:
        print("数据目录不存在，请先准备数据集")
