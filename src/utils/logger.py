import logging
import os
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from collections import defaultdict
from torchvision.utils import make_grid

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = defaultdict(list)
        self.writer = SummaryWriter(log_dir)
        self.logger = logging.getLogger(__name__)
        
        # 设置日志格式
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # 添加步骤计数器
        self.step = 0
        
    def log_training(self, epoch, loss, psnr, ssim, lr):
        """记录训练信息"""
        self.writer.add_scalar('Loss/train', loss, epoch)
        self.writer.add_scalar('Metrics/PSNR', psnr, epoch)
        self.writer.add_scalar('Metrics/SSIM', ssim, epoch)
        self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        self.logger.info(
            f'Epoch [{epoch}] Loss: {loss:.4f} PSNR: {psnr:.2f} SSIM: {ssim:.4f} LR: {lr:.6f}'
        )
        
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
        
    def log_metrics(self, epoch, metrics):
        """记录指标"""
        for name, value in metrics.items():
            self.metrics[name].append(value)
            self.writer.add_scalar(name, value, epoch)
    
    def log_metrics_step(self, metrics):
        """记录每个训练步骤的指标"""
        for name, value in metrics.items():
            self.writer.add_scalar(f'Step/{name}', value, self.step)
        self.step += 1
            
    def get_metric(self, name):
        """获取指标历史记录"""
        return self.metrics[name]
        
    def log_images(self, epoch, images_dict):
        """记录图像"""
        for name, images in images_dict.items():
            grid = make_grid(images, normalize=True)
            self.writer.add_image(name, grid, epoch)
            
    def close(self):
        """关闭writer"""
        self.writer.close()