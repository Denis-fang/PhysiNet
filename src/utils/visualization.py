import matplotlib.pyplot as plt
import numpy as np
import torch

def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    反归一化张量，将其从归一化空间转换回[0,1]范围
    
    参数:
    - tensor: 归一化后的张量
    - mean: 归一化时使用的均值
    - std: 归一化时使用的标准差
    
    返回:
    - 反归一化后的张量，范围在[0, 1]
    """
    # 克隆张量以避免修改原始数据
    tensor = tensor.clone().detach()
    
    # 反归一化步骤：
    # 1. 首先乘以标准差
    # 2. 然后加上均值
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # 确保值在[0,1]范围内
    return torch.clamp(tensor, 0, 1)

def plot_results(original, corrupted, reconstructed, save_path=None):
    """显示原始图像、损坏图像、重建图像和误差图的对比"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 反归一化图像
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    # 反归一化并转换为numpy数组
    original = denormalize(original, mean, std).cpu().numpy().transpose(1, 2, 0)
    corrupted = denormalize(corrupted, mean, std).cpu().numpy().transpose(1, 2, 0)
    reconstructed = denormalize(reconstructed, mean, std).cpu().numpy().transpose(1, 2, 0)
    
    # 计算误差图
    error = np.abs(original - reconstructed)
    error_map = np.mean(error, axis=2)  # 转换为灰度图
    
    # 显示原始图像
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 显示损坏的图像
    axes[1].imshow(corrupted)
    axes[1].set_title('Corrupted')
    axes[1].axis('off')
    
    # 显示重建图像
    axes[2].imshow(reconstructed)
    axes[2].set_title('Reconstructed')
    axes[2].axis('off')
    
    # 显示误差图
    im = axes[3].imshow(error_map, cmap='jet')
    axes[3].set_title('Error Map')
    axes[3].axis('off')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label('Error')
    
    # 确保所有子图大小一致
    for ax in axes:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 