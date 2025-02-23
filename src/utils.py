import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def plot_results(original, reconstructed, save_path=None):
    """显示原始图像和重建图像的对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # 转换张量为numpy数组
    original = original.detach().cpu().numpy().transpose(1, 2, 0)
    reconstructed = reconstructed.detach().cpu().numpy().transpose(1, 2, 0)
    
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(reconstructed)
    ax2.set_title('Reconstructed')
    ax2.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_psnr(original, reconstructed):
    """计算峰值信噪比（PSNR）"""
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(original, reconstructed):
    """计算结构相似性指数（SSIM）"""
    # TODO: 实现SSIM计算
    pass 