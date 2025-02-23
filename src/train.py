import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import PhysiNet
from data_loader import ImageDataset
import yaml
from tqdm import tqdm

def train(config_path='experiments/config/train_config.yaml'):
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化模型
    model = PhysiNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 准备数据
    train_dataset = ImageDataset(config['data_path'], mode='train')
    train_loader = DataLoader(train_dataset, 
                            batch_size=config['batch_size'],
                            shuffle=True)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch}: Loss = {loss.item():.4f}')
        
    # 保存模型
    torch.save(model.state_dict(), config['save_path'])

if __name__ == '__main__':
    train() 