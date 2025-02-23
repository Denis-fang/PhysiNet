import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import cv2
import logging

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

def calculate_ssim(img1, img2):
    """计算结构相似性指数（SSIM）"""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 