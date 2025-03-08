import torch
import torch.nn as nn
from .mamba_block import MambaBlock

class MixtureOfMamba(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=3, d_model=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.d_model = d_model
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            MambaPhysiNet(in_channels, out_channels, d_model=d_model)  # 使用完整的d_model
            for _ in range(num_experts)
        ])
        
        # 创建路由网络
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=1)  # 确保权重和为1
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 计算路由权重
        weights = self.router(x)  # [B, num_experts]
        
        # 专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        # 加权组合专家输出
        output = torch.zeros_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):
            weight = weights[:, i].view(B, 1, 1, 1)
            output += weight * expert_output
        
        return output 