import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from src.models import MambaPhysiNet, SepONet, PDENet
from src.data_loaders import get_dataloaders
from src.utils import calculate_psnr, calculate_ssim, plot_results
import yaml
from tqdm import tqdm
import os
import gc
from torchvision import models
from src.utils import Logger
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models import get_model
from src.utils.checkpoint import CheckpointManager
from src.eval import Evaluator
from src.utils.visualize import Visualizer
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import logging

class SSIMLoss(nn.Module):
    """SSIM损失函数"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def forward(self, img1, img2):
        # 确保输入在[0,1]范围内
        img1 = (img1 + 1) / 2.0
        img2 = (img2 + 1) / 2.0
        
        # 检查输入是否包含NaN
        if torch.isnan(img1).any() or torch.isnan(img2).any():
            return torch.tensor(1.0, device=img1.device)  # 返回最大损失
            
        (_, channel, height, width) = img1.size()

        if channel == self.channel and self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            
        window = self.window
        if channel != self.channel:
            window = self._create_window(self.window_size, channel).to(img1.device)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()  # 转换为损失（1-SSIM）
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)  # 转换为损失（1-SSIM）

class Trainer:
    def __init__(self, config_path='experiments/config/train_config.yaml'):
        # 加载配置
        self.config_path = config_path
        self.config = self._load_config()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        self.logger = Logger(self.config['training']['log_dir'])
        
        # 初始化数据加载器
        self.train_loader, self.test_loader = self._init_data_loaders()
        
        # 初始化模型
        self.model = self._init_model()
        
        # 初始化优化器
        self.optimizer = self._init_optimizer()
        
        # 初始化学习率调度器
        self.scheduler = self._init_scheduler()
        
        # 初始化损失函数
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = self._get_perceptual_loss()
        self.ssim_loss = SSIMLoss()
        
        # 设置损失权重
        self.mse_weight = 1.0
        self.perceptual_weight = 0.05
        self.ssim_weight = 0.1
        self.gcdd_weight = self.config['model'].get('gcdd_weight', 0.0)
        
        # 记录损失权重配置
        self.logger.logger.info(f"损失权重配置: MSE={self.mse_weight}, 感知={self.perceptual_weight}, SSIM={self.ssim_weight}, GCDD={self.gcdd_weight}")
        
        # 梯度累积步数
        self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        # 梯度裁剪值
        self.grad_clip = self.config['training'].get('grad_clip', 1.0)
        
        # 混合精度训练设置
        self.mixed_precision = self.config['training'].get('mixed_precision', {}).get('enabled', False)
        
        # 强制使用float32类型进行混合精度训练
        self.precision_dtype = torch.float32
        
        # 初始化梯度缩放器，用于混合精度训练
        self.scaler = GradScaler(
            enabled=self.mixed_precision,
            init_scale=2.**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )
        
        # 创建必要的目录
        os.makedirs(self.config['training']['save_dir'], exist_ok=True)
        os.makedirs(self.config['training']['log_dir'], exist_ok=True)
        
        # 初始化检查点管理器
        self.checkpoint_manager = CheckpointManager(
            self.config['training']['save_dir']
        )
        
        # 初始化评估器
        self.evaluator = Evaluator(
            self.model,
            self.test_loader,
            self.device
        )
        
        # 添加可视化器
        self.visualizer = Visualizer(self.config)
        
        # 添加性能指标跟踪
        self.best_metrics = {
            'psnr': 0,
            'ssim': 0,
            'loss': float('inf')
        }
        
    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
        
    def _init_data_loaders(self):
        return get_dataloaders(self.config)
        
    def _init_model(self):
        return get_model(self.config).to(self.device)
        
    def _init_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
    def _init_scheduler(self):
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=float(self.config['training']['scheduler']['min_lr'])
        )
        
    def _get_perceptual_loss(self):
        vgg = models.vgg16(pretrained=True)
        layers = [4, 9, 16, 23]
        vgg_features = []
        for i in range(max(layers) + 1):
            vgg_features.append(vgg.features[i])
            
        vgg_model = nn.Sequential(*vgg_features)
        for param in vgg_model.parameters():
            param.requires_grad = False
            
        return vgg_model.to(self.device)
        
    def _compute_perceptual_loss(self, output, target):
        """计算感知损失"""
        if not hasattr(self, 'vgg'):
            self.vgg = self._get_perceptual_loss()
        
        # 确保输入在[0,1]范围内
        output = torch.clamp(output.float(), 0, 1)
        target = torch.clamp(target.float(), 0, 1)
        
        # 将输入转换为VGG期望的格式
        output = output * 255.0
        target = target * 255.0
        
        # 获取VGG特征
        output_features = []
        target_features = []
        
        with torch.no_grad():
            x_out = output
            x_target = target
            
            for i, layer in enumerate(self.vgg):
                x_out = layer(x_out)
                x_target = layer(x_target)
                
                if i in [2, 7, 14, 21]:  # 选择特定的层
                    output_features.append(x_out)
                    target_features.append(x_target)
        
        loss = 0.0
        # 调整权重，增加浅层特征的权重，提高颜色还原度
        weights = [0.2, 0.3, 0.3, 0.2]  # 增加浅层特征的权重
        
        for i in range(len(output_features)):
            loss += weights[i] * F.mse_loss(output_features[i], target_features[i])
        
        return loss
        
    def _compute_loss(self, output, target):
        mse_loss = F.mse_loss(output, target)
        
        perceptual_loss = self._compute_perceptual_loss(output, target)
        
        ssim_loss = 1 - self._compute_ssim_loss(output, target)
        
        # 调整损失权重，增加颜色相关损失的权重
        loss_weights = {
            'mse': 0.4,        # 增加MSE权重，提高颜色还原度
            'perceptual': 0.3, # 减小感知损失权重
            'ssim': 0.3        # 保持SSIM权重不变
        }
        
        total_loss = (
            loss_weights['mse'] * mse_loss +
            loss_weights['perceptual'] * perceptual_loss +
            loss_weights['ssim'] * ssim_loss
        )
        
        return total_loss

    def _compute_ssim_loss(self, output, target):
        output = output.float()
        target = target.float()
        
        output = torch.clamp(output.float(), 0, 1)
        target = torch.clamp(target.float(), 0, 1)
        
        from pytorch_ssim import ssim
        
        batch_ssim = ssim(output, target, window_size=11)
        
        return batch_ssim.mean()
        
    def combined_loss(self, output, target):
        if torch.isnan(output).any() or torch.isinf(output).any():
            self.logger.logger.error(f"损失计算错误: 输出值范围异常")
            return None
        
        if torch.isnan(target).any() or torch.isinf(target).any():
            self.logger.logger.error(f"损失计算错误: 目标值范围异常")
            return None
        
        output = output.float()
        target = target.float()
        
        output = torch.clamp(output.float(), -10.0, 10.0)
        
        try:
            mse_loss = self.mse_loss(output, target)
            
            if torch.isnan(mse_loss).any() or torch.isinf(mse_loss).any():
                self.logger.logger.error(f"MSE损失计算错误: 值为{mse_loss.item()}")
                return None
            
            ssim_loss = self.ssim_loss(output, target)
            
            if torch.isnan(ssim_loss).any() or torch.isinf(ssim_loss).any():
                self.logger.logger.error(f"SSIM损失计算错误: 值为{ssim_loss.item()}")
                ssim_loss = torch.tensor(0.0, device=output.device)
            
            try:
                perceptual_loss = self._compute_perceptual_loss(output, target)
                
                if torch.isnan(perceptual_loss).any() or torch.isinf(perceptual_loss).any():
                    self.logger.logger.error(f"感知损失计算错误: 值为{perceptual_loss.item()}")
                    perceptual_loss = torch.tensor(0.0, device=output.device)
            except Exception as e:
                self.logger.logger.error(f"感知损失计算异常: {str(e)}")
                perceptual_loss = torch.tensor(0.0, device=output.device)
            
            gcdd_loss = 0.0
            mamba_loss = 0.0
            if hasattr(self.model, 'compute_gcdd_loss'):
                gcdd_loss = self.model.compute_gcdd_loss(output, target)
                
                if torch.isnan(gcdd_loss).any() or torch.isinf(gcdd_loss).any():
                    self.logger.logger.error(f"GCDD损失计算错误: 值为{gcdd_loss.item()}")
                    gcdd_loss = 0.0
                    
            if hasattr(self.model, 'compute_mamba_diversity_loss'):
                mamba_loss = self.model.compute_mamba_diversity_loss()
                
                if torch.isnan(mamba_loss).any() or torch.isinf(mamba_loss).any():
                    self.logger.logger.error(f"MixOfMamba损失计算错误: 值为{mamba_loss.item()}")
                    mamba_loss = 0.0
            
            total_loss = (
                self.mse_weight * mse_loss + 
                self.perceptual_weight * perceptual_loss + 
                self.ssim_weight * ssim_loss + 
                self.gcdd_weight * gcdd_loss
            )
            
            self.logger.log_metrics_step({
                'loss/mse': mse_loss.item(),
                'loss/perceptual': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else 0.0,
                'loss/ssim': ssim_loss.item(),
                'loss/gcdd': gcdd_loss if isinstance(gcdd_loss, float) else gcdd_loss.item(),
                'loss/mamba': mamba_loss if isinstance(mamba_loss, float) else mamba_loss.item(),
                'loss/total': total_loss.item()
            })
            
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                self.logger.logger.error(f"总损失计算错误: 值为{total_loss.item()}")
                return mse_loss
            
            return total_loss
        
        except Exception as e:
            self.logger.logger.error(f"损失计算异常: {str(e)}")
            return None
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        valid_batches = 0
        running_psnr = 0
        running_ssim = 0
        
        pbar = tqdm(self.train_loader)
        
        for batch_idx, (data, _) in enumerate(pbar):
            try:
                data = data.float().to(self.device)
                
                if batch_idx % self.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
                
                if data.max() > 1.0 or data.min() < 0.0:
                    data = data / 255.0 if data.max() > 1.0 else data
                
                corrupted = self.corrupt_image(
                    data,
                    corruption_type=self.config['data']['corruption']['type'],
                    noise_level=self.config['data']['corruption']['noise_level'],
                    mask_ratio=self.config['data']['corruption']['mask_ratio']
                )
                
                with autocast(enabled=self.mixed_precision, dtype=self.precision_dtype):
                    output = self.model(corrupted)
                    
                    output = output.float()
                    
                    loss = self.combined_loss(output, data)
                
                if loss is None or torch.isnan(loss):
                    self.logger.logger.warning(f"检测到nan损失，跳过此批次")
                    continue
                
                loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.grad_clip
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                
                with torch.no_grad():
                    psnr = calculate_psnr(output, data)
                    ssim = calculate_ssim(output, data)
                    
                    if not (torch.isnan(torch.tensor(psnr)) or torch.isnan(torch.tensor(ssim))):
                        running_psnr += psnr
                        running_ssim += ssim
                        valid_batches += 1
                
                pbar.set_description(
                    f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f} PSNR: {psnr:.2f} SSIM: {ssim:.4f}"
                )
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                
                if self.config.get('memory_optimization', {}).get('empty_cache', False) and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                self.logger.logger.error(f"训练错误: {str(e)}")
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else float('nan')
        avg_psnr = running_psnr / valid_batches if valid_batches > 0 else float('nan')
        avg_ssim = running_ssim / valid_batches if valid_batches > 0 else float('nan')
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'loss': avg_loss,
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
        
    def validate(self):
        self.model.eval()
        total_psnr = 0
        total_ssim = 0
        valid_samples = 0
        
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                
                # 确保数据在合理范围内
                data = torch.clamp(data, 0, 1)
                output = torch.clamp(output, 0, 1)
                
                try:
                    psnr = calculate_psnr(data, output)
                    ssim = calculate_ssim(data, output)
                    
                    # 检查计算结果是否有效
                    if not (np.isnan(psnr) or np.isinf(psnr) or np.isnan(ssim) or np.isinf(ssim)):
                        total_psnr += psnr
                        total_ssim += ssim
                        valid_samples += 1
                except Exception as e:
                    print(f"验证过程中出错: {e}")
                    continue
        
        # 避免除零错误
        avg_psnr = total_psnr / max(valid_samples, 1)
        avg_ssim = total_ssim / max(valid_samples, 1)
        
        return avg_psnr, avg_ssim
        
    def train(self):
        for epoch in range(self.config['training']['epochs']):
            train_metrics = self.train_epoch()
            
            val_metrics = self.validate()
            
            self.scheduler.step()
            
            self.logger.log_training(
                epoch,
                train_metrics['loss'],
                train_metrics['psnr'],
                train_metrics['ssim'],
                self.optimizer.param_groups[0]['lr']
            )
            
            self.visualize_epoch(epoch)
            
            if (epoch + 1) % self.config['training']['checkpoint']['save_freq'] == 0:
                self.checkpoint_manager.save(
                    self.model,
                    epoch,
                    train_metrics['loss'],
                    train_metrics['psnr'],
                    train_metrics['ssim']
                )

    def corrupt_image(self, image, corruption_type='gaussian', noise_level=0.1, mask_ratio=0.9):
        """
        对图像进行损坏处理
        
        参数:
            image: 输入图像
            corruption_type: 损坏类型
            noise_level: 噪声级别
            mask_ratio: 掩码比例
        """
        # 确保图像在[0,1]范围内
        image = torch.clamp(image, 0, 1)
        
        # 创建损坏图像的副本
        corrupted = image.clone()
        
        if corruption_type == 'gaussian':
            # 添加高斯噪声
            noise = torch.randn_like(image) * noise_level
            corrupted = image + noise
            corrupted = torch.clamp(corrupted, 0, 1)
        
        elif corruption_type == 'mask':
            # 随机掩码
            mask = torch.rand_like(image) > (1 - mask_ratio)
            corrupted = image * mask.float()
        
        elif corruption_type == 'mixed':
            # 混合损坏：先添加噪声，然后应用掩码
            # 添加高斯噪声
            noise = torch.randn_like(image) * noise_level
            corrupted = image + noise
            
            # 随机掩码，但保留更多的原始像素
            mask = torch.rand_like(image) > (1 - mask_ratio)
            
            # 应用掩码，保留原始图像的大部分内容
            corrupted = corrupted * mask.float() + image * (1 - mask.float())
            
            # 确保值在[0,1]范围内
            corrupted = torch.clamp(corrupted, 0, 1)
            
            # 添加颜色保持步骤，保持原始图像的颜色分布
            # 计算每个通道的平均值
            original_mean = torch.mean(image, dim=[2, 3], keepdim=True)
            corrupted_mean = torch.mean(corrupted, dim=[2, 3], keepdim=True)
            
            # 调整损坏图像的颜色分布
            corrupted = corrupted + 0.5 * (original_mean - corrupted_mean)
            corrupted = torch.clamp(corrupted, 0, 1)
        
        elif corruption_type == 'paper_mixed':
            # 论文中使用的混合损坏
            # 添加高斯噪声
            noise = torch.randn_like(image) * noise_level
            corrupted = image + noise
            
            # 随机掩码
            mask = torch.rand_like(image) > (1 - mask_ratio)
            corrupted = corrupted * mask.float()
            
            # 确保值在[0,1]范围内
            corrupted = torch.clamp(corrupted, 0, 1)
        
        else:
            raise ValueError(f"不支持的损坏类型: {corruption_type}")
        
        return corrupted

    def save_reconstructions(self, epoch):
        self.model.eval()
        with torch.no_grad():
            data = next(iter(self.test_loader))[0][:4].to(self.device)
            
            corrupted = self.corrupt_image(
                data,
                corruption_type=self.config['data']['corruption']['type'],
                noise_level=self.config['data']['corruption']['noise_level'],
                mask_ratio=self.config['data']['corruption']['mask_ratio']
            )
            
            output = self.model(corrupted)
            
            save_dir = os.path.join(self.config['training']['save_dir'], 'reconstructions')
            os.makedirs(save_dir, exist_ok=True)
            
            for i in range(min(4, len(data))):
                save_path = os.path.join(
                    save_dir,
                    f'epoch_{epoch}_sample_{i}.png'
                )
                plot_results(data[i], corrupted[i], output[i], save_path)
                
            if hasattr(self, 'logger'):
                self.logger.log_images(epoch, {
                    'original': data,
                    'corrupted': corrupted,
                    'reconstructed': output
                })

    def train_gcdd(self, data, target):
        output = self.model(data)
        
        gcdd_loss = self.model.gcdd.loss_function(output, target)
        
        reg_loss = 0
        for param in self.model.parameters():
            reg_loss += torch.norm(param, p=2)
        
        total_loss = (
            gcdd_loss +
            self.config['training']['lambda_reg'] * reg_loss
        )
        
        return total_loss, output
        
    def train_seponet(self, data):
        attention_loss = ...
        reconstruction_loss = ...
        return self.mse_loss(output, target) + attention_loss + reconstruction_loss
        
    def train_mixture_mamba(self, data):
        diversity_loss = ...
        routing_loss = ...
        return self.mse_loss(output, target) + diversity_loss + routing_loss

    def save_checkpoint(self, epoch, metrics, is_final=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metrics': self.best_metrics,
            'config': self.config
        }
        
        if not is_final:
            self.checkpoint_manager.save_checkpoint(
                state,
                is_best=metrics['psnr'] > self.best_metrics['psnr'],
                filename=f'checkpoint_epoch_{epoch}.pth'
            )
            
            if metrics['psnr'] > self.best_metrics['psnr']:
                self.best_metrics['psnr'] = metrics['psnr']
                self.best_metrics['ssim'] = metrics['ssim']
                self.logger.logger.info(
                    f'New best model saved! PSNR: {metrics["psnr"]:.2f}, SSIM: {metrics["ssim"]:.4f}'
                )
        else:
            self.checkpoint_manager.save_checkpoint(
                state,
                is_best=False,
                filename='final_model.pth'
            )
            self.logger.logger.info('Final model saved!')

    def visualize_epoch(self, epoch):
        try:
            self.model.eval()
            with torch.no_grad():
                try:
                    test_batch = next(iter(self.test_loader))
                    data = test_batch[0][:self.config['training']['visualization']['num_samples']].to(self.device)
                except Exception as e:
                    self.logger.logger.warning(f"获取测试数据失败: {str(e)}")
                    return
                
                try:
                    corrupted = self.corrupt_image(
                        data,
                        corruption_type=self.config['data']['corruption']['type'],
                        noise_level=self.config['data']['corruption']['noise_level'],
                        mask_ratio=self.config['data']['corruption']['mask_ratio']
                    )
                except Exception as e:
                    self.logger.logger.warning(f"图像损坏处理失败: {str(e)}")
                    return
                
                try:
                    output = self.model(corrupted)
                except Exception as e:
                    self.logger.logger.error(f"模型重建失败: {str(e)}")
                    return
                
                try:
                    psnr = calculate_psnr(output, data)
                    ssim = calculate_ssim(output, data)
                except Exception as e:
                    self.logger.logger.warning(f"指标计算失败: {str(e)}")
                    psnr = 0
                    ssim = 0
                
                if self.config['training']['visualization']['enabled']:
                    save_dir = os.path.join(self.config['training']['save_dir'], 'visualizations', f'epoch_{epoch}')
                    os.makedirs(save_dir, exist_ok=True)
                    
                    try:
                        for i in range(len(data)):
                            fig = self.visualizer.visualize_results(
                                data[i], corrupted[i], output[i]
                            )
                            plt.savefig(os.path.join(save_dir, f'sample_{i}.png'), dpi=300)
                            plt.close(fig)
                            
                            self._save_grayscale_view(
                                data[i], corrupted[i], output[i],
                                os.path.join(save_dir, f'sample_{i}_gray.png')
                            )
                    except Exception as e:
                        self.logger.logger.error(f"可视化保存失败: {str(e)}")
                
                if hasattr(self, 'logger'):
                    try:
                        self.logger.log_images(
                            epoch,
                            {
                                'original': data,
                                'corrupted': corrupted,
                                'reconstructed': output
                            }
                        )
                        self.logger.log_metrics(
                            epoch,
                            {
                                'vis/psnr': psnr,
                                'vis/ssim': ssim
                            }
                        )
                    except Exception as e:
                        self.logger.logger.warning(f"日志记录失败: {str(e)}")
        
        except Exception as e:
            self.logger.logger.error(f"可视化错误: {str(e)}")
        
    def _save_grayscale_view(self, original, corrupted, reconstructed, save_path):
        try:
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(original.device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(original.device)
            
            original_denorm = original.clone()
            corrupted_denorm = corrupted.clone()
            reconstructed_denorm = reconstructed.clone()
            
            original_denorm = original_denorm * std + mean
            corrupted_denorm = corrupted_denorm * std + mean
            reconstructed_denorm = reconstructed_denorm * std + mean
            
            original_denorm = torch.clamp(original_denorm, 0, 1)
            corrupted_denorm = torch.clamp(corrupted_denorm, 0, 1)
            reconstructed_denorm = torch.clamp(reconstructed_denorm, 0, 1)
            
            original_gray = original_denorm.mean(dim=0, keepdim=True)
            corrupted_gray = corrupted_denorm.mean(dim=0, keepdim=True)
            reconstructed_gray = reconstructed_denorm.mean(dim=0, keepdim=True)
            
            original_np = original_gray.squeeze().cpu().numpy()
            corrupted_np = corrupted_gray.squeeze().cpu().numpy()
            reconstructed_np = reconstructed_gray.squeeze().cpu().numpy()
            
            error = np.abs(original_np - reconstructed_np)
            
            fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)
            
            axes[0].imshow(original_np, cmap='gray')
            axes[0].set_title('Original', fontsize=14)
            axes[0].axis('off')
            
            axes[1].imshow(corrupted_np, cmap='gray')
            axes[1].set_title('Corrupted', fontsize=14)
            axes[1].axis('off')
            
            axes[2].imshow(reconstructed_np, cmap='gray')
            axes[2].set_title('Reconstructed', fontsize=14)
            axes[2].axis('off')
            
            im = axes[3].imshow(error, cmap='jet', vmin=0, vmax=0.5)
            axes[3].set_title('Error Map', fontsize=14)
            axes[3].axis('off')
            
            divider = make_axes_locatable(axes[3])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            self.logger.logger.error(f"保存灰度视图失败: {str(e)}")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train() 