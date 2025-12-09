"""
UNet 预测脚本
用于对新图像进行建筑物分割预测
"""
import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.unet_model import UNet


def predict_image(net, image_path, device, scale=1.0, out_threshold=0.5):
    """
    对单张图像进行预测
    
    参数:
        net: 训练好的模型
        image_path: 输入图像路径
        device: 设备
        scale: 图像缩放比例
        out_threshold: 输出阈值
    
    返回:
        预测的掩码
    """
    net.eval()
    
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    
    # 缩放图像
    if scale != 1.0:
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img = img.resize((new_width, new_height), Image.BILINEAR)
    
    # 转换为tensor
    img_array = np.array(img)
    img_tensor = torch.from_numpy(img_array).type(torch.FloatTensor)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
    img_tensor = img_tensor / 255.0  # 先归一化到0-1
        
    # ImageNet标准化（与训练时一致）
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    # 移动到设备
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    
    # 预测
    with torch.no_grad():
        output = net(img_tensor)
        
        if net.n_classes > 1:
            # 多分类 - 取建筑物类别（第1个通道）的概率
            probs = F.softmax(output, dim=1)[0]
            mask = (probs[1] > out_threshold).float()  # 取第1类（建筑物）
        else:
            # 二分类
            probs = torch.sigmoid(output)[0]
            mask = (probs > out_threshold).float()
    
    # 转换为numpy数组
    mask = mask.cpu().squeeze().numpy()
    
    return mask, img_array


def mask_to_image(mask):
    """
    将掩码转换为PIL图像
    
    参数:
        mask: 预测的掩码数组
    
    返回:
        PIL图像
    """
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def visualize_prediction(original_img, mask, save_path=None):
    """
    可视化预测结果
    
    参数:
        original_img: 原始图像
        mask: 预测掩码
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('原始图像', fontsize=14)
    axes[0].axis('off')
    
    # 预测掩码
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('预测掩码', fontsize=14)
    axes[1].axis('off')
    
    # 叠加显示
    overlay = original_img.copy()
    mask_rgb = np.zeros_like(original_img)
    mask_rgb[:, :, 0] = mask * 255  # 红色通道显示建筑物
    overlay = cv2.addWeighted(overlay, 0.7, mask_rgb.astype(np.uint8), 0.3, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('叠加结果', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'可视化结果已保存至: {save_path}')
    
    plt.show()


def batch_predict(net, input_dir, output_dir, device, scale=1.0, out_threshold=0.5, visualize=False):
    """
    批量预测目录中的所有图像
    
    参数:
        net: 训练好的模型
        input_dir: 输入图像目录
        output_dir: 输出目录
        device: 设备
        scale: 图像缩放比例
        out_threshold: 输出阈值
        visualize: 是否生成可视化结果
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    print(f'找到 {len(image_files)} 张图像，开始预测...')
    
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        
        # 预测
        mask, original_img = predict_image(net, img_path, device, scale, out_threshold)
        
        # 保存掩码
        mask_img = mask_to_image(mask)
        output_path = os.path.join(output_dir, f'mask_{img_file}')
        mask_img.save(output_path)
        
        print(f'已处理: {img_file} -> {output_path}')
        
        # 可视化
        if visualize:
            vis_path = os.path.join(output_dir, f'vis_{img_file}')
            visualize_prediction(original_img, mask, vis_path)


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用训练好的模型进行建筑物分割预测')
    
    parser.add_argument('--model', '-m', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入图像或目录路径')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出目录路径')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='图像缩放比例')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='预测阈值')
    parser.add_argument('--classes', '-c', type=int, default=2, help='分类数量')
    parser.add_argument('--bilinear', action='store_true', default=False, help='使用双线性插值')
    parser.add_argument('--visualize', '-v', action='store_true', default=False, help='生成可视化结果')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(device=device)
    
    print(f'从 {args.model} 加载模型...')
    state_dict = torch.load(args.model, map_location=device)
    
    # 处理不同格式的检查点
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    print('模型加载成功！')
    
    # 判断输入是文件还是目录
    if os.path.isfile(args.input):
        # 单张图像预测
        print(f'预测单张图像: {args.input}')
        mask, original_img = predict_image(model, args.input, device, args.scale, args.threshold)
        
        # 保存结果
        Path(args.output).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(args.output, 'prediction_mask.png')
        mask_img = mask_to_image(mask)
        mask_img.save(output_path)
        print(f'预测结果已保存至: {output_path}')
        
        # 可视化
        if args.visualize:
            try:
                import cv2
                vis_path = os.path.join(args.output, 'prediction_visualization.png')
                visualize_prediction(original_img, mask, vis_path)
            except ImportError:
                print('警告: 需要安装 opencv-python 才能使用可视化功能')
                print('可以通过 pip install opencv-python 安装')
    
    elif os.path.isdir(args.input):
        # 批量预测
        print(f'批量预测目录: {args.input}')
        batch_predict(model, args.input, args.output, device, args.scale, args.threshold, args.visualize)
    
    else:
        print(f'错误: 输入路径 {args.input} 不存在')
    
    print('预测完成！')
