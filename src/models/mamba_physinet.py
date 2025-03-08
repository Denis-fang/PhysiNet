import torch
import torch.nn as nn
import torch.nn.functional as F
from .mixture_of_mamba import MixtureOfMamba, MambaBlock
from .gcdd_layer import GCDDLayer
from src.args import ModelArgs

class MambaPhysiNet(nn.Module):
    def __init__(self, in_channels, out_channels=None, d_model=64, d_state=16, d_conv=4, expand=2, config=None):
        super().__init__()
        
        out_channels = out_channels or in_channels
        self.d_model = d_model
        
        # 保存配置
        self.config = config or {
            'model': {
                'gcdd': {
                    'time_steps': 100,
                    'dt': 0.1,
                    'alpha': 1.0,
                    'beta': 0.5
                },
                'mamba': {
                    'num_experts': 3,
                    'd_state': 16,
                    'd_conv': 4,
                    'expand': 2
                }
            }
        }
        
        # 创建配置对象
        self.args = ModelArgs()
        
        # 获取图像尺寸
        self.image_size = config['data']['image_size']  # 64
        
        # 计算特征图尺寸变化
        self.num_encoder_layers = 3  # 编码器层数
        self.feature_sizes = []
        h = w = self.image_size  # 现在是64x64
        for _ in range(self.num_encoder_layers):
            self.feature_sizes.append((h, w))
            h, w = h//2, w//2  # 每层下采样2倍
        
        # 优化编码器通道数设计，逐层增加特征
        enc_channels = [in_channels, d_model, d_model*2, d_model*4]
        
        # 创建编码器
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_channels[i], enc_channels[i+1], 3, padding=1),
                nn.BatchNorm2d(enc_channels[i+1]),
                nn.LeakyReLU(0.2)
            ) for i in range(len(enc_channels)-1)
        ])
        
        # 优化GCDD参数
        gcdd_config = self.config['model']['gcdd']
        self.gcdd = GCDDLayer(
            time_steps=gcdd_config.get('time_steps', 100),
            dt=gcdd_config.get('dt', 0.1),
            alpha=gcdd_config.get('alpha', 1.0),
            beta=gcdd_config.get('beta', 0.5)
        )
        
        # 优化解码器通道数设计
        dec_channels = [d_model*4, d_model*2, d_model, 32]
        
        # 创建解码器
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dec_channels[i] * 2, dec_channels[i+1], 3, padding=1),
                nn.BatchNorm2d(dec_channels[i+1]) if i < len(dec_channels)-2 else nn.Identity(),
                nn.LeakyReLU(0.2) if i < len(dec_channels)-2 else nn.ReLU()
            ) for i in range(len(dec_channels)-1)
        ])
        
        # 简化最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),  # 添加批归一化
            nn.Sigmoid()  # 使用Sigmoid确保输出在[0,1]范围内
        )
        
        # 创建MixtureOfMamba
        mamba_config = self.config['model']['mamba']
        self.mamba_mixture = MixtureOfMamba(
            in_channels=d_model*4,  # bottleneck的通道数
            out_channels=d_model*4,
            num_experts=mamba_config.get('num_experts', 3),
            d_model=d_model*4,
            d_state=mamba_config.get('d_state', 16),
            d_conv=mamba_config.get('d_conv', 4),
            expand=mamba_config.get('expand', 2)
        )
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W == self.image_size, f"输入图像尺寸{H}x{W}与配置的尺寸{self.image_size}x{self.image_size}不匹配"
        
        # 存储编码器特征
        features = []
        
        # 编码器前向传播
        for enc in self.encoder:
            x = enc(x)
            features.append(x)
            x = self.pool(x)
        
        # 重塑为序列形式并使用MixtureOfMamba
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_seq = self.mamba_mixture(x_seq)  # 处理序列
        x = x_seq.transpose(1, 2).reshape(B, -1, H, W)  # 重塑回特征图
        
        # 应用物理约束
        x = self.gcdd(x)
        
        # 解码器前向传播
        for i, dec in enumerate(self.decoder):
            x = self.upsample(x)
            x = torch.cat([x, features[-(i+1)]], dim=1)
            x = dec(x)
        
        # 最终输出处理
        x = self.final_conv(x)
        
        return x 