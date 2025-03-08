import torch
import torch.nn as nn
import torch.nn.functional as F

class GCDDLayer(nn.Module):
    """增强版Gaussian Curvature-Driven Diffusion Layer"""
    def __init__(self, time_steps=10, dt=0.01, alpha=0.1, beta=0.01):
        super().__init__()
        self.time_steps = time_steps
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        
        # 添加可学习参数
        self.alpha_param = nn.Parameter(torch.tensor(alpha))
        self.beta_param = nn.Parameter(torch.tensor(beta))
        
        # 添加卷积核用于梯度计算
        self.sobel_x = nn.Parameter(
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float() / 8,
            requires_grad=False
        )
        self.sobel_y = nn.Parameter(
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float() / 8,
            requires_grad=False
        )
        
    def compute_gaussian_curvature(self, u):
        """优化的高斯曲率计算"""
        # 添加数值稳定性
        eps = 1e-6
        
        # 计算一阶和二阶导数
        ux = self.gradient_x(u)
        uy = self.gradient_y(u)
        uxx = self.gradient_x(ux)
        uxy = self.gradient_y(ux)
        uyy = self.gradient_y(uy)
        
        # 添加梯度值裁剪
        ux = torch.clamp(ux, -10, 10)
        uy = torch.clamp(uy, -10, 10)
        uxx = torch.clamp(uxx, -10, 10)
        uxy = torch.clamp(uxy, -10, 10)
        uyy = torch.clamp(uyy, -10, 10)
        
        # 计算高斯曲率时添加数值稳定性
        denominator = torch.clamp(1 + ux**2 + uy**2, min=eps)
        K = (uxx*uyy - uxy**2) / (denominator**2 + eps)
        
        # 裁剪曲率值
        K = torch.clamp(K, -5, 5)
        return K
        
    def gradient_x(self, u):
        """优化的x方向梯度计算"""
        kernel = self.sobel_x.view(1, 1, 3, 3).repeat(u.size(1), 1, 1, 1)
        return F.conv2d(u, kernel, padding=1, groups=u.size(1))
        
    def gradient_y(self, u):
        """优化的y方向梯度计算"""
        kernel = self.sobel_y.view(1, 1, 3, 3).repeat(u.size(1), 1, 1, 1)
        return F.conv2d(u, kernel, padding=1, groups=u.size(1))
        
    def mean_curvature(self, u):
        """优化的平均曲率计算"""
        ux = self.gradient_x(u)
        uy = self.gradient_y(u)
        uxx = self.gradient_x(ux)
        uxy = self.gradient_y(ux)
        uyy = self.gradient_y(uy)
        
        # 裁剪梯度值
        ux = torch.clamp(ux, -10, 10)
        uy = torch.clamp(uy, -10, 10)
        uxx = torch.clamp(uxx, -10, 10)
        uxy = torch.clamp(uxy, -10, 10)
        uyy = torch.clamp(uyy, -10, 10)
        
        # 添加数值稳定性
        eps = 1e-6
        denominator = torch.sqrt(1 + ux**2 + uy**2 + eps)
        H = ((1 + uy**2)*uxx - 2*ux*uy*uxy + (1 + ux**2)*uyy) / (2 * denominator**3 + eps)
        
        # 裁剪曲率值
        H = torch.clamp(H, -5, 5)
        return H
        
    def forward(self, x):
        """前向传播 - 增强版GCDD"""
        u = x
        
        # 使用可学习参数
        alpha = torch.abs(self.alpha_param)  # 确保为正值
        beta = torch.abs(self.beta_param)    # 确保为正值
        
        # 保存初始输入
        u_original = u.clone()
        
        # 迭代扩散过程
        for _ in range(self.time_steps):
            # 计算高斯曲率和平均曲率
            K = self.compute_gaussian_curvature(u)
            H = self.mean_curvature(u)
            
            # 计算扩散项
            diffusion_term = alpha * K + beta * H
            
            # 裁剪扩散项
            diffusion_term = torch.clamp(diffusion_term, -1, 1)
            
            # 更新u
            u = u + self.dt * diffusion_term
            
            # 添加边界条件
            u = F.pad(u[:, :, 1:-1, 1:-1], (1, 1, 1, 1), mode='replicate')
            
            # 检查数值稳定性
            if torch.isnan(u).any() or torch.isinf(u).any():
                print("警告: GCDD迭代中检测到NaN或Inf值")
                u = torch.nan_to_num(u, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 添加残差连接，增强稳定性
        u = 0.7 * u + 0.3 * u_original
        
        return u
        
    def loss_function(self, pred, target):
        """增强的GCDD特定损失函数"""
        # 重建损失
        mse_loss = F.mse_loss(pred, target)
        
        # 曲率正则化
        K_pred = self.compute_gaussian_curvature(pred)
        K_target = self.compute_gaussian_curvature(target)
        curvature_loss = F.mse_loss(K_pred, K_target)
        
        # 梯度正则化
        grad_pred_x = self.gradient_x(pred)
        grad_pred_y = self.gradient_y(pred)
        grad_target_x = self.gradient_x(target)
        grad_target_y = self.gradient_y(target)
        
        gradient_loss = (F.mse_loss(grad_pred_x, grad_target_x) + 
                        F.mse_loss(grad_pred_y, grad_target_y))
        
        # 使用可学习参数
        alpha = torch.abs(self.alpha_param)
        beta = torch.abs(self.beta_param)
        
        # 组合损失
        total_loss = mse_loss + alpha * curvature_loss + beta * gradient_loss
        
        # 检查数值稳定性
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("警告: GCDD损失计算中检测到NaN或Inf值")
            return mse_loss
            
        return total_loss 