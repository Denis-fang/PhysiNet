import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置后端
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid, save_image
import cv2
import logging
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def plot_results(original, reconstructed, save_path=None):
    """显示原始图像和重建图像的对比"""
    # 使用torchvision的save_image替代matplotlib
    from torchvision.utils import save_image
    
    # 将图像拼接在一起
    images = torch.stack([original, reconstructed], dim=0)
    save_image(images, save_path, nrow=2, normalize=True, range=(-1, 1))

def calculate_psnr(original, reconstructed):
    """计算峰值信噪比（PSNR）"""
    # 确保输入是在CPU上的numpy数组并归一化到[0,1]
    original = (original.detach().cpu().numpy() + 1) / 2.0
    reconstructed = (reconstructed.detach().cpu().numpy() + 1) / 2.0
    
    # 转换维度顺序从[B,C,H,W]到[B,H,W,C]
    original = np.transpose(original, (0, 2, 3, 1))
    reconstructed = np.transpose(reconstructed, (0, 2, 3, 1))
    
    # 计算每个batch的PSNR
    batch_size = original.shape[0]
    psnr_sum = 0
    
    for i in range(batch_size):
        mse = np.mean((original[i] - reconstructed[i]) ** 2)
        if mse < 1e-10:  # 避免log(0)
            psnr_sum += 50  # 降低最大值到50dB
            continue
            
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        psnr_sum += min(50, psnr)  # 限制最大值为50dB
        
    return psnr_sum / batch_size

def calculate_ssim(img1, img2):
    """计算结构相似性指数（SSIM）"""
    # 确保输入是在CPU上的numpy数组并归一化到[0,1]
    img1 = (img1.detach().cpu().numpy() + 1) / 2.0  # 如果数据范围是[-1,1]
    img2 = (img2.detach().cpu().numpy() + 1) / 2.0
    
    # 转换维度顺序从[B,C,H,W]到[B,H,W,C]
    img1 = np.transpose(img1, (0, 2, 3, 1))
    img2 = np.transpose(img2, (0, 2, 3, 1))
    
    # 将数据范围调整到[0,255]用于SSIM计算
    img1 = (img1 * 255).astype(np.float32)
    img2 = (img2 * 255).astype(np.float32)
    
    # 初始化SSIM值
    ssim_value = 0
    batch_size = img1.shape[0]
    
    # 对每个batch分别计算SSIM
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

class Logger:
    def __init__(self, log_dir='logs'):
        # 创建日志目录
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(log_dir, current_time)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置文件日志
        self.logger = logging.getLogger('PhysiNet')
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler(os.path.join(self.log_dir, 'train.log'))
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
    def log_training(self, epoch, train_loss, psnr, ssim, lr):
        """记录训练信息"""
        self.logger.info(
            f'Epoch {epoch}: Loss={train_loss:.4f}, PSNR={psnr:.2f}, '
            f'SSIM={ssim:.4f}, LR={lr:.6f}'
        )
        
        # 记录到TensorBoard
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Metrics/PSNR', psnr, epoch)
        self.writer.add_scalar('Metrics/SSIM', ssim, epoch)
        self.writer.add_scalar('LR', lr, epoch)
        
    def log_validation(self, epoch, val_loss, val_psnr, val_ssim):
        """记录验证信息"""
        self.logger.info(
            f'Validation: Loss={val_loss:.4f}, PSNR={val_psnr:.2f}, '
            f'SSIM={val_ssim:.4f}'
        )
        
        # 记录到TensorBoard
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Metrics/val_PSNR', val_psnr, epoch)
        self.writer.add_scalar('Metrics/val_SSIM', val_ssim, epoch)
        
    def log_test(self, test_psnr, test_ssim):
        """记录测试信息"""
        self.logger.info(
            f'Test Results: PSNR={test_psnr:.2f}, SSIM={test_ssim:.4f}'
        )
        
    def log_images(self, epoch, images_dict):
        """记录图像到TensorBoard"""
        for name, img in images_dict.items():
            self.writer.add_images(name, img, epoch)
            
    def close(self):
        """关闭日志记录器"""
        self.writer.close() 