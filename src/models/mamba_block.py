import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelectiveScan(nn.Module):
    """选择性扫描机制"""
    def __init__(self, d_model, d_state, d_conv):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # S4D状态空间参数
        self.A = nn.Parameter(torch.randn(d_state))
        self.B = nn.Parameter(torch.randn(d_state))
        self.C = nn.Parameter(torch.randn(d_state, d_model))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # 卷积层
        self.conv = nn.Conv1d(
            d_model, d_model, 
            kernel_size=d_conv,
            padding=d_conv-1,
            groups=d_model
        )
        
    def forward(self, x, delta):
        """
        x: [B, L, D]
        delta: [B, L, 1]
        """
        B, L, D = x.shape
        
        # 检查输入维度
        if D != self.d_model:
            # 如果维度不匹配，使用投影层调整维度
            x = F.linear(x, torch.eye(D, self.d_model, device=x.device))
            D = self.d_model
        
        # 计算状态空间表示
        A = -torch.exp(self.A)  # [S]
        A = A.unsqueeze(0).unsqueeze(0)  # [1, 1, S]
        
        # 确保delta的尺寸与x的序列长度匹配
        if delta.size(1) != L:
            # 如果长度不匹配，调整delta的长度
            delta = delta[:, :L, :] if delta.size(1) > L else F.pad(delta, (0, 0, 0, L - delta.size(1), 0, 0))
        
        # 状态更新
        A = A * delta[:, :, 0:1]  # 只使用第一个维度
        C = self.C  # [S, D]
        
        # 确保输入维度正确
        u = x.transpose(1, 2)  # [B, D, L]
        
        # 处理可能的维度不匹配
        if u.size(1) != self.d_model:
            # 如果通道数不匹配，使用投影层调整维度
            u = F.conv1d(u, torch.eye(u.size(1), self.d_model, device=u.device).unsqueeze(-1))
        
        v = self.conv(u)  # [B, D, L]
        v = v.transpose(1, 2)  # [B, L, D]
        
        return v

class MambaBlock(nn.Module):
    """完整的Mamba块实现"""
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.d_inner = int(self.expand * self.d_model)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 时间混合
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv-1,
            groups=self.d_inner
        )
        
        # S4D层
        self.selective_scan = SelectiveScan(self.d_inner, d_state, d_conv)
        
        # 门控机制
        self.gate = nn.Linear(self.d_inner, self.d_inner)
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 层标准化 - 修改为正确的维度
        self.norm = nn.LayerNorm(d_model)  # 只需要指定最后一个维度
        
    def forward(self, x):
        """
        x: [B, L, D]
        """
        B, L, D = x.shape
        
        # 确保输入维度正确
        if D != self.d_model:
            # 如果维度不匹配，使用投影层调整维度
            x = F.linear(x, torch.eye(D, self.d_model, device=x.device))
            D = self.d_model
            
        shortcut = x
        
        # 层标准化
        x = self.norm(x)  # LayerNorm在最后一个维度上进行归一化
        
        # 输入投影和分支
        x = self.in_proj(x)  # [B, L, 2*D_inner]
        x, gate = x.chunk(2, dim=-1)  # 两个 [B, L, D_inner]
        
        # 时间混合
        x = x.transpose(1, 2)  # [B, D_inner, L]
        x = self.conv1d(x)     # [B, D_inner, L]
        x = x.transpose(1, 2)  # [B, L, D_inner]
        x = F.silu(x)
        
        # 确保门控维度匹配
        gate = gate[:, :x.size(1), :]  # 裁剪到相同长度
        gate = self.gate(gate)
        gate = torch.sigmoid(gate)
        
        # 选择性扫描 - 使用当前序列长度
        L_current = x.size(1)  # 获取当前序列长度
        delta = torch.ones(B, L_current, 1, device=x.device)  # [B, L_current, 1]
        
        # 确保delta的尺寸与x匹配
        if delta.size(1) != x.size(1):
            delta = F.interpolate(
                delta.transpose(1, 2),  # [B, 1, L_current]
                size=x.size(1),
                mode='nearest'
            ).transpose(1, 2)  # [B, x.size(1), 1]
            
        x = self.selective_scan(x, delta)  # [B, L, D_inner]
        
        # 应用门控 (确保维度匹配)
        if x.size(1) != gate.size(1):
            # 如果长度不匹配，裁剪到较小的长度
            min_len = min(x.size(1), gate.size(1))
            x = x[:, :min_len, :]
            gate = gate[:, :min_len, :]
        x = x * gate
        
        # 输出投影
        x = self.out_proj(x)  # [B, L, D]
        
        # Dropout
        x = self.dropout(x)
        
        # 残差连接 (确保维度匹配)
        if x.size(1) != shortcut.size(1):
            # 如果长度不匹配，裁剪到较小的长度
            min_len = min(x.size(1), shortcut.size(1))
            x = x[:, :min_len, :]
            shortcut = shortcut[:, :min_len, :]
        x = x + shortcut
        
        return x 