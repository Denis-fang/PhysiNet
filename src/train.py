import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import PhysiNet
from data_loader import get_dataloaders
from utils import calculate_psnr, calculate_ssim, plot_results
import yaml
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, config_path='experiments/config/train_config.yaml'):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = PhysiNet(
            in_channels=self.config['model']['in_channels'],
            out_channels=self.config['model']['out_channels']
        ).to(self.device)
        
        # 获取数据加载器
        self.train_loader, self.test_loader = get_dataloaders(
            self.config['data']['dataset'],
            self.config['training']['batch_size'],
            self.config['data']['data_path']
        )
        
        # 定义损失函数
        self.criterion = nn.MSELoss()
        
        # 定义优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs']
        )
        
        # 创建保存目录
        os.makedirs(self.config['training']['save_dir'], exist_ok=True)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(self.train_loader)):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, data)  # 自重建任务
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
        
    def validate(self):
        self.model.eval()
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                
                # 计算评估指标
                psnr = calculate_psnr(data, output)
                ssim = calculate_ssim(data, output)
                
                total_psnr += psnr
                total_ssim += ssim
                
        avg_psnr = total_psnr / len(self.test_loader)
        avg_ssim = total_ssim / len(self.test_loader)
        
        return avg_psnr, avg_ssim
        
    def train(self):
        best_psnr = 0
        
        for epoch in range(self.config['training']['epochs']):
            # 训练一个epoch
            train_loss = self.train_epoch()
            
            # 验证
            psnr, ssim = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存最佳模型
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.config['training']['save_dir'], 'best_model.pth')
                )
                
            # 打印进度
            print(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}:')
            print(f'Loss: {train_loss:.4f}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}')
            
if __name__ == '__main__':
    trainer = Trainer()
    trainer.train() 