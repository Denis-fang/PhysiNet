import torch
import torch.nn as nn
import torch.nn.functional as F

class PDENet(nn.Module):
    """Gaussian Curvature-Driven Diffusion Network"""
    def __init__(self, in_channels, out_channels, time_steps=100, diffusion_coeff=0.1):
        super().__init__()
        self.time_steps = time_steps
        self.diffusion_coeff = diffusion_coeff
        
        # 空间梯度算子
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = self.sobel_x.transpose(2, 3)
        
        # 编码器-解码器结构
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
        
    def compute_gaussian_curvature(self, u):
        """计算高斯曲率"""
        # 计算一阶和二阶导数
        ux = F.conv2d(u, self.sobel_x.to(u.device), padding=1)
        uy = F.conv2d(u, self.sobel_y.to(u.device), padding=1)
        uxx = F.conv2d(ux, self.sobel_x.to(u.device), padding=1)
        uxy = F.conv2d(ux, self.sobel_y.to(u.device), padding=1)
        uyy = F.conv2d(uy, self.sobel_y.to(u.device), padding=1)
        
        # 计算高斯曲率 K = (uxx*uyy - uxy^2)/(1 + ux^2 + uy^2)^2
        numerator = uxx * uyy - uxy.pow(2)
        denominator = (1 + ux.pow(2) + uy.pow(2)).pow(2)
        return numerator / (denominator + 1e-6)
        
    def forward(self, x):
        # 编码
        features = self.encoder(x)
        
        # PDE求解
        u = features
        for _ in range(self.time_steps):
            # 计算高斯曲率
            G = self.compute_gaussian_curvature(u)
            
            # 计算扩散项
            diff_term = self.diffusion_coeff * G
            
            # 更新u
            u = u + diff_term
            
        # 解码
        out = self.decoder(u)
        return out

    def loss_function(self, u, theta, P, Q, G):
        """
        计算损失函数 L = ||e1||^2 + ||e2||^2 + ||e3||^2 + ||e4||^2 + ||e5||^2
        """
        # 计算各项误差
        e1 = u.diff(dim=0) - theta * (P + Q)  # ut - θ(u)(Px + Qy)
        e2 = theta - (1 + u.pow(2).sum(dim=1) + u.pow(2).sum(dim=2)).pow(1.5)
        e3 = P - self.compute_diffusion_coefficient(G) * u.diff(dim=1)  # P - φ(G)ux
        e4 = Q - self.compute_diffusion_coefficient(G) * u.diff(dim=2)  # Q - φ(G)uy
        e5 = G - self.compute_gaussian_curvature(u)
        
        # 计算总损失
        loss = (e1.pow(2).mean() + e2.pow(2).mean() + 
                e3.pow(2).mean() + e4.pow(2).mean() + 
                e5.pow(2).mean())
                
        return loss 