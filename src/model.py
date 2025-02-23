import torch
import torch.nn as nn
import torch.nn.functional as F

class SepONet(nn.Module):
    """可分离算子网络实现"""
    def __init__(self, in_channels, out_channels):
        super(SepONet, self).__init__()
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, 
                                    kernel_size=3, padding=1, groups=in_channels)
        self.point_conv = nn.Conv2d(out_channels, out_channels, 
                                   kernel_size=1)
        
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x

class GCDDLayer(nn.Module):
    """基于高斯曲率驱动扩散的物理约束层"""
    def __init__(self):
        super(GCDDLayer, self).__init__()
        self.theta = nn.Parameter(torch.ones(1))
        
    def compute_gaussian_curvature(self, u):
        """计算高斯曲率 G"""
        # 计算一阶和二阶导数
        ux = F.conv2d(u, self.sobel_x(), padding=1)
        uy = F.conv2d(u, self.sobel_y(), padding=1)
        uxx = F.conv2d(ux, self.sobel_x(), padding=1)
        uxy = F.conv2d(ux, self.sobel_y(), padding=1)
        uyy = F.conv2d(uy, self.sobel_y(), padding=1)
        
        # 计算高斯曲率
        denominator = (1 + ux**2 + uy**2)**2
        G = (uxx*uyy - uxy**2) / (denominator + 1e-6)
        return G
        
    def forward(self, u):
        """实现G-CDD方程的前向传播"""
        G = self.compute_gaussian_curvature(u)
        theta_u = self.theta * u
        
        # 计算扩散系数
        phi_G = torch.exp(-torch.abs(G))
        
        # 计算扩散项
        P = phi_G * F.conv2d(u, self.sobel_x(), padding=1)
        Q = phi_G * F.conv2d(u, self.sobel_y(), padding=1)
        
        # 计算发散
        div_term = F.conv2d(P, self.sobel_x(), padding=1) + \
                  F.conv2d(Q, self.sobel_y(), padding=1)
                  
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
    def __init__(self, in_channels=3, out_channels=3):
        super(PhysiNet, self).__init__()
        
        # SepONet编码器
        self.encoder = nn.ModuleList([
            SepONet(in_channels, 64),
            SepONet(64, 128),
            SepONet(128, 256)
        ])
        
        # G-CDD物理约束层
        self.gcdd = GCDDLayer()
        
        # SepONet解码器
        self.decoder = nn.ModuleList([
            SepONet(256, 128),
            SepONet(128, 64),
            SepONet(64, out_channels)
        ])
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
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
            x = x + features[-(i+1)]  # 跳跃连接
            x = dec(x)
            
        return x 