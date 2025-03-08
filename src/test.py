import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录到路径

import torch
import torch.nn as nn
from models.mamba_physinet import MambaPhysiNet
from data_loader import get_dataloaders
from utils import calculate_psnr, calculate_ssim, plot_results
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

class Tester:
    def __init__(self, config_path='experiments/config/test_config.yaml'):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型 - 改用MambaPhysiNet
        self.model = MambaPhysiNet(
            in_channels=self.config['model']['in_channels'],
            out_channels=self.config['model']['out_channels'],
            d_model=self.config['model']['d_model'],
            d_state=self.config['model']['d_state'],
            d_conv=self.config['model']['d_conv'],
            expand=self.config['model']['expand']
        ).to(self.device)
        
        # 加载预训练模型
        checkpoint = torch.load(self.config['testing']['model_path'])
        self.model.load_state_dict(checkpoint['model_state_dict'])  # 注意这里要用['model_state_dict']
        
        # 获取测试数据
        _, self.test_loader = get_dataloaders(
            self.config['data']['dataset'],
            self.config['testing']['batch_size'],
            self.config['data']['data_path']
        )
        
        # 创建结果保存目录
        self.save_dir = self.config['testing']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
    def test(self):
        self.model.eval()
        total_psnr = 0
        total_ssim = 0
        
        # 用于可视化的结果
        results = []
        
        with torch.no_grad():
            for idx, (data, _) in enumerate(tqdm(self.test_loader)):
                data = data.to(self.device)
                output = self.model(data)
                
                # 确保输出和输入的通道数相同
                assert output.shape[1] == data.shape[1], \
                    f"Output channels ({output.shape[1]}) != Input channels ({data.shape[1]})"
                
                # 计算评估指标
                psnr = calculate_psnr(data, output)
                ssim = calculate_ssim(data, output)
                
                total_psnr += psnr
                total_ssim += ssim
                
                # 保存一些结果用于可视化
                if idx < 5:  # 只保存前5个batch的结果
                    results.append({
                        'input': data[0].cpu(),
                        'output': output[0].cpu(),
                        'target': data[0].cpu()
                    })
                
                # 保存重建结果
                if idx < self.config['testing']['num_save_samples']:
                    plot_results(
                        data[0], output[0],
                        os.path.join(self.save_dir, f'sample_{idx}.png')
                    )
                    
        avg_psnr = total_psnr / len(self.test_loader)
        avg_ssim = total_ssim / len(self.test_loader)
        
        print(f'测试结果：PSNR = {avg_psnr:.2f}, SSIM = {avg_ssim:.4f}')
        
        # 可视化结果
        self.visualize_results(results)
        
    def visualize_results(self, results):
        """可视化重建结果"""
        for i, result in enumerate(results):
            plt.figure(figsize=(15, 5))
            
            # 显示输入图像
            plt.subplot(131)
            plt.imshow(result['input'].permute(1,2,0))
            plt.title('Input')
            
            # 显示重建结果
            plt.subplot(132)
            plt.imshow(result['output'].permute(1,2,0))
            plt.title('Reconstruction')
            
            # 显示目标图像
            plt.subplot(133)
            plt.imshow(result['target'].permute(1,2,0))
            plt.title('Target')
            
            plt.savefig(os.path.join(self.save_dir, f'reconstruction_{i}.png'))
            plt.close()

if __name__ == '__main__':
    tester = Tester()
    tester.test() 