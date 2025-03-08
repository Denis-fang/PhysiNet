import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops.selective_scan import selective_scan_fn

class BaselineDenseMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # 投影层 - 修改输入维度
        self.in_proj = nn.Linear(d_model, self.d_inner)  # 从d_model投影到d_inner
        
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM参数
        # 修改A的初始化
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.expand(self.d_inner, -1).contiguous()  # [d_inner, d_state]
        self.A = nn.Parameter(torch.log(A))
        self.A._no_weight_decay = True
        
        # 修改B和C的初始化
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        
        # D和dt保持不变
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        """
        x: (batch, d_model)
        """
        # 投影到内部维度
        x = self.in_proj(x)  # [B, d_inner]
        
        # 添加序列维度用于卷积
        x = x.unsqueeze(-1)  # [B, d_inner, 1]
        
        # 卷积处理
        x_conv = self.conv1d(x)  # [B, d_inner, 1]
        
        # SSM处理
        x_ssm = x  # [B, d_inner, 1]
        
        # 准备SSM参数
        A = -torch.exp(self.A)  # [d_inner, d_state]
        B = self.B.unsqueeze(0)  # [1, d_inner, d_state]
        C = self.C.unsqueeze(0)  # [1, d_inner, d_state]
        
        # 应用SSM
        x_ssm = selective_scan_fn(
            x_ssm,  # [B, d_inner, 1]
            self.dt,  # [d_inner]
            A,  # [d_inner, d_state]
            B,  # [1, d_inner, d_state]
            C,  # [1, d_inner, d_state]
            self.D,  # [d_inner]
            delta_bias=None,
            delta_softplus=True
        )  # [B, d_inner, 1]
        
        # 组合并激活
        x = F.silu(x_conv[:, :, :1] + x_ssm)  # [B, d_inner, 1]
        x = x.squeeze(-1)  # [B, d_inner]
        
        # 输出投影
        x = self.out_proj(x)  # [B, d_model]
        
        return x 