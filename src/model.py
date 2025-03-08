import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SepONetWithSE(nn.Module):
    """带SE模块的可分离算子网络"""
    def __init__(self, in_channels, out_channels, se_reduction=16):
        super(SepONetWithSE, self).__init__()
        
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, 
                                    kernel_size=3, padding=1, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)  # 添加BN
        
        self.point_conv = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 添加BN
        
        self.se = SELayer(out_channels, reduction=se_reduction)
        self.relu = nn.ReLU(inplace=True)
        
        # 添加残差连接
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.point_conv(x)
        x = self.bn2(x)
        
        x = self.se(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        x += identity  # 残差连接
        x = self.relu(x)
        
        return x

class GCDDLayer(nn.Module):
    """基于高斯曲率驱动扩散的物理约束层"""
    def __init__(self):
        super(GCDDLayer, self).__init__()
        self.theta = nn.Parameter(torch.ones(1))
        
    def compute_gaussian_curvature(self, u):
        """计算高斯曲率 G"""
        # 获取输入通道数
        batch_size, channels, height, width = u.shape
        
        # 创建适应多通道的Sobel算子
        sobel_x = self.sobel_x().to(u.device)
        sobel_y = self.sobel_y().to(u.device)
        
        # 扩展Sobel算子以匹配输入通道数
        sobel_x = sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(channels, 1, 1, 1)
        
        # 计算一阶和二阶导数 (分组卷积)
        ux = F.conv2d(u, sobel_x, padding=1, groups=channels)
        uy = F.conv2d(u, sobel_y, padding=1, groups=channels)
        uxx = F.conv2d(ux, sobel_x, padding=1, groups=channels)
        uxy = F.conv2d(ux, sobel_y, padding=1, groups=channels)
        uyy = F.conv2d(uy, sobel_y, padding=1, groups=channels)
        
        # 计算高斯曲率
        denominator = (1 + ux**2 + uy**2)**2
        G = (uxx*uyy - uxy**2) / (denominator + 1e-6)
        return G
        
    def forward(self, u):
        """实现G-CDD方程的前向传播"""
        G = self.compute_gaussian_curvature(u)
        theta_u = self.theta * u
        
        # 获取输入通道数
        channels = u.shape[1]
        
        # 扩展Sobel算子
        sobel_x = self.sobel_x().to(u.device).repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y().to(u.device).repeat(channels, 1, 1, 1)
        
        # 计算扩散系数
        phi_G = torch.exp(-torch.abs(G))
        
        # 计算扩散项 (使用分组卷积)
        P = phi_G * F.conv2d(u, sobel_x, padding=1, groups=channels)
        Q = phi_G * F.conv2d(u, sobel_y, padding=1, groups=channels)
        
        # 计算发散
        div_term = F.conv2d(P, sobel_x, padding=1, groups=channels) + \
                  F.conv2d(Q, sobel_y, padding=1, groups=channels)
                  
        return u + div_term
    
    def sobel_x(self):
        """Sobel算子 x方向"""
        kernel = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3)
        return kernel
        
    def sobel_y(self):
        """Sobel算子 y方向"""
        kernel = torch.tensor([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3)
        return kernel

class PhysiNet(nn.Module):
    def __init__(self, in_channels, out_channels=None, se_reduction=16):
        super(PhysiNet, self).__init__()
        
        # 如果没有指定输出通道数，则与输入通道数相同
        out_channels = out_channels or in_channels
        
        # 编码器通道数
        enc_channels = [in_channels, 64, 128, 256]
        
        # 解码器通道数 (确保最后输出与输入通道数相同)
        dec_channels = [256, 128, 64, out_channels]
        
        # SepONet编码器 (带SE模块)
        self.encoder = nn.ModuleList([
            SepONetWithSE(enc_channels[i], enc_channels[i+1], se_reduction)
            for i in range(len(enc_channels)-1)
        ])
        
        # G-CDD物理约束层
        self.gcdd = GCDDLayer()
        
        # SepONet解码器 (带SE模块)
        self.decoder = nn.ModuleList([
            SepONetWithSE(dec_channels[i], dec_channels[i+1], se_reduction)
            for i in range(len(dec_channels)-1)
        ])
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 添加全局SE模块
        self.global_se = SELayer(out_channels, reduction=se_reduction)
        
    def forward(self, x):
        # 编码过程
        features = []
        for enc in self.encoder:
            x = enc(x)
            features.append(x)
            x = self.pool(x)
            
        # 应用物理约束
        x = self.gcdd(x)
        
        # 解码过程
        for i, dec in enumerate(self.decoder):
            x = self.upsample(x)
            # 添加跳跃连接
            if i < len(features):
                x = x + features[-(i+1)]
            x = dec(x)
            
        # 应用全局SE注意力
        x = self.global_se(x)
            
        return x 