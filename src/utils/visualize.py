import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import torch.nn.functional as F

class Visualizer:
    def __init__(self, config):
        self.config = config
        
    def plot_training_progress(self, metrics):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
        
        # Plot loss
        if 'train_loss' in metrics:
            axes[0, 0].plot(metrics['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in metrics:
            axes[0, 0].plot(metrics['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=16)
        axes[0, 0].set_xlabel('Epoch', fontsize=14)
        axes[0, 0].set_ylabel('Loss', fontsize=14)
        axes[0, 0].legend(fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot PSNR
        if 'psnr' in metrics:
            axes[0, 1].plot(metrics['psnr'], linewidth=2, color='green')
        axes[0, 1].set_title('PSNR', fontsize=16)
        axes[0, 1].set_xlabel('Epoch', fontsize=14)
        axes[0, 1].set_ylabel('PSNR (dB)', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot SSIM
        if 'ssim' in metrics:
            axes[1, 0].plot(metrics['ssim'], linewidth=2, color='orange')
        axes[1, 0].set_title('SSIM', fontsize=16)
        axes[1, 0].set_xlabel('Epoch', fontsize=14)
        axes[1, 0].set_ylabel('SSIM', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot learning rate
        if 'lr' in metrics:
            axes[1, 1].plot(metrics['lr'], linewidth=2, color='purple')
        axes[1, 1].set_title('Learning Rate', fontsize=16)
        axes[1, 1].set_xlabel('Epoch', fontsize=14)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def visualize_results(self, original, corrupted, reconstructed):
        """Visualize original image, corrupted image, reconstructed image and error map"""
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)
        
        # 使用BRAST2021数据集的标准化参数
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(original.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(original.device)
        
        # 复制张量以避免修改原始数据
        original_denorm = original.clone()
        corrupted_denorm = corrupted.clone()
        reconstructed_denorm = reconstructed.clone()
        
        # 反标准化处理
        original_denorm = original_denorm * std + mean
        corrupted_denorm = corrupted_denorm * std + mean
        reconstructed_denorm = reconstructed_denorm * std + mean
        
        # 确保值在[0,1]范围内
        original_denorm = torch.clamp(original_denorm, 0, 1)
        corrupted_denorm = torch.clamp(corrupted_denorm, 0, 1)
        reconstructed_denorm = torch.clamp(reconstructed_denorm, 0, 1)
        
        # 转换为numpy数组
        original_np = self._tensor_to_numpy(original_denorm)
        corrupted_np = self._tensor_to_numpy(corrupted_denorm)
        reconstructed_np = self._tensor_to_numpy(reconstructed_denorm)
        
        # 计算误差图
        error = np.abs(original_np - reconstructed_np).mean(axis=2)
        
        # 绘制图像
        axes[0].imshow(original_np)
        axes[0].set_title('Original', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(corrupted_np)
        axes[1].set_title('Corrupted', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(reconstructed_np)
        axes[2].set_title('Reconstructed', fontsize=14)
        axes[2].axis('off')
        
        # 添加误差图
        im = axes[3].imshow(error, cmap='jet', vmin=0, vmax=0.5)
        axes[3].set_title('Error Map', fontsize=14)
        axes[3].axis('off')
        divider = make_axes_locatable(axes[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        plt.tight_layout()
        return fig
        
    def _is_bgr(self, tensor):
        """检查是否为BGR格式"""
        if tensor.shape[0] != 3:
            return False
        # 计算每个通道的平均值
        means = tensor.mean(dim=(1,2))
        # BGR格式通常B通道均值大于R通道
        return means[0] > means[2]
        
    def _color_correction(self, target, source):
        """改进的颜色校正算法"""
        # 1. 计算每个通道的均值和标准差
        t_mean = target.mean(dim=(1,2), keepdim=True)
        t_std = target.std(dim=(1,2), keepdim=True)
        s_mean = source.mean(dim=(1,2), keepdim=True)
        s_std = source.std(dim=(1,2), keepdim=True)
        
        # 2. 应用颜色匹配
        normalized = (source - s_mean) / (s_std + 1e-6)
        matched = normalized * t_std + t_mean
        
        # 3. 添加局部对比度增强
        kernel_size = 3
        padding = kernel_size // 2
        local_mean = F.avg_pool2d(matched, kernel_size, stride=1, padding=padding)
        local_std = torch.sqrt(F.avg_pool2d(matched**2, kernel_size, stride=1, padding=padding) - local_mean**2 + 1e-6)
        enhanced = (matched - local_mean) / (local_std + 1e-6) * t_std + t_mean
        
        # 4. 融合结果
        alpha = 0.7  # 控制增强强度
        result = alpha * matched + (1 - alpha) * enhanced
        
        return torch.clamp(result, 0, 1)
        
    def visualize_gcdd(self, before_gcdd, after_gcdd, curvature, save_path=None):
        """Visualize GCDD effects"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
        
        # Convert to numpy arrays
        before_gcdd = self._tensor_to_numpy(before_gcdd)
        after_gcdd = self._tensor_to_numpy(after_gcdd)
        curvature = self._tensor_to_numpy(curvature, is_single_channel=True)
        
        # Plot images
        axes[0].imshow(before_gcdd)
        axes[0].set_title('Before GCDD', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(after_gcdd)
        axes[1].set_title('After GCDD', fontsize=14)
        axes[1].axis('off')
        
        # Display curvature as heatmap
        im = axes[2].imshow(curvature, cmap='jet')
        axes[2].set_title('Gaussian Curvature', fontsize=14)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            
        return fig
        
    def visualize_seponet(self, features, attention_maps):
        """Visualize SepONet attention maps"""
        pass
        
    def visualize_mixture_mamba(self, expert_outputs, routing_weights, save_path=None):
        """Visualize MixOfMamba effects"""
        num_experts = len(expert_outputs)
        fig, axes = plt.subplots(1, num_experts + 2, figsize=(6 * (num_experts + 2), 6), dpi=150)
        
        # Plot each expert's output
        for i, expert_output in enumerate(expert_outputs):
            expert_img = self._tensor_to_numpy(expert_output)
            axes[i].imshow(expert_img)
            axes[i].set_title(f'Expert {i+1} Output', fontsize=14)
            axes[i].axis('off')
        
        # Plot routing weights
        weights_img = routing_weights.cpu().numpy()
        bar_positions = np.arange(len(weights_img))
        axes[num_experts].bar(bar_positions, weights_img)
        axes[num_experts].set_title('Routing Weights', fontsize=14)
        axes[num_experts].set_xticks(bar_positions)
        axes[num_experts].set_xticklabels([f'Expert {i+1}' for i in range(len(weights_img))], fontsize=12)
        
        # Plot weighted fusion result
        weighted_sum = torch.zeros_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):
            weighted_sum += expert_output * routing_weights[i]
        
        weighted_img = self._tensor_to_numpy(weighted_sum)
        axes[num_experts + 1].imshow(weighted_img)
        axes[num_experts + 1].set_title('Weighted Fusion Result', fontsize=14)
        axes[num_experts + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            
        return fig
        
    def _tensor_to_numpy(self, tensor, is_single_channel=False):
        """Convert tensor to numpy array"""
        # Ensure it's a CPU tensor
        tensor = tensor.detach().cpu()
        
        if is_single_channel:
            # Single channel image
            img = tensor.numpy()
            return img
        else:
            # Multi-channel image
            if tensor.dim() == 4:
                tensor = tensor[0]  # Take the first sample
            
            # 检查通道数
            if tensor.shape[0] != 3:
                if tensor.shape[0] == 1:
                    # 单通道 -> 三通道
                    tensor = tensor.repeat(3, 1, 1)
                elif tensor.shape[0] > 3:
                    # 多通道 -> 三通道 (只取前三个通道)
                    tensor = tensor[:3]
            
            # Convert to HWC format
            img = tensor.permute(1, 2, 0).numpy()
            
            # Clip to [0,1] range
            img = np.clip(img, 0, 1)
            
            return img 