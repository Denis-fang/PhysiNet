import torch
import torch.nn as nn
from model import PhysiNet
from data_loader import get_dataloaders
from utils import calculate_psnr, calculate_ssim, plot_results
import yaml
import os
from tqdm import tqdm

class Tester:
    def __init__(self, config_path='experiments/config/test_config.yaml'):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = PhysiNet(
            in_channels=self.config['model']['in_channels'],
            out_channels=self.config['model']['out_channels']
        ).to(self.device)
        
        # 加载预训练模型
        checkpoint = torch.load(self.config['testing']['model_path'])
        self.model.load_state_dict(checkpoint)
        
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
        
        with torch.no_grad():
            for idx, (data, _) in enumerate(tqdm(self.test_loader)):
                data = data.to(self.device)
                output = self.model(data)
                
                # 计算评估指标
                psnr = calculate_psnr(data, output)
                ssim = calculate_ssim(data, output)
                
                total_psnr += psnr
                total_ssim += ssim
                
                # 保存重建结果
                if idx < self.config['testing']['num_save_samples']:
                    plot_results(
                        data[0], output[0],
                        os.path.join(self.save_dir, f'sample_{idx}.png')
                    )
                    
        avg_psnr = total_psnr / len(self.test_loader)
        avg_ssim = total_ssim / len(self.test_loader)
        
        print(f'测试结果：PSNR = {avg_psnr:.2f}, SSIM = {avg_ssim:.4f}')
        
if __name__ == '__main__':
    tester = Tester()
    tester.test() 