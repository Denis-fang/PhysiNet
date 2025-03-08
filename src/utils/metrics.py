import torch
import numpy as np
import cv2

def calculate_psnr(original, reconstructed):
    """计算峰值信噪比（PSNR）"""
    # 检查输入是否包含NaN或inf
    if torch.isnan(original).any() or torch.isnan(reconstructed).any() or \
       torch.isinf(original).any() or torch.isinf(reconstructed).any():
        return 0.0
    
    # 确保输入在[0,1]范围内
    original = torch.clamp(original, 0, 1)
    reconstructed = torch.clamp(reconstructed, 0, 1)
    
    mse = torch.mean((original - reconstructed) ** 2)
    # 添加一个小的epsilon值以避免除零
    epsilon = 1e-10
    if mse < epsilon:
        return 100.0  # 返回一个较大但有限的值，而不是无穷大
    
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse + epsilon))
    
    # 限制PSNR的范围，避免极端值
    return float(torch.clamp(psnr, 0, 100).item())

def calculate_ssim(img1, img2):
    """计算结构相似性指数（SSIM）"""
    # 确保输入在[0,1]范围内
    img1 = torch.clamp(img1.detach().cpu(), 0, 1).numpy()
    img2 = torch.clamp(img2.detach().cpu(), 0, 1).numpy()
    
    img1 = np.transpose(img1, (0, 2, 3, 1))
    img2 = np.transpose(img2, (0, 2, 3, 1))
    
    # 转换到[0,255]范围
    img1 = (img1 * 255).astype(np.float32)
    img2 = (img2 * 255).astype(np.float32)
    
    ssim_value = 0
    batch_size = img1.shape[0]
    
    for i in range(batch_size):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        mu1 = cv2.filter2D(img1[i], -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2[i], -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(img1[i]**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2[i]**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1[i] * img2[i], -1, window)[5:-5, 5:-5] - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        ssim_value += ssim_map.mean()
    
    return ssim_value / batch_size 