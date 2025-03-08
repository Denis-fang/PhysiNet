import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def ssm_kernel(
    A: torch.Tensor,  # (H, N)
    B: torch.Tensor,  # (H, N)
    C: torch.Tensor,  # (H, N)
    D: torch.Tensor,  # (H,)
    u: torch.Tensor,  # (B, H, L)
    delta: torch.Tensor,  # (B, H, L)
    z: torch.Tensor = None,  # (B, H, L)
):
    """
    实现SSM核心计算
    """
    # 获取维度
    B, H, L = u.shape
    N = A.shape[1]
    
    # 扩展维度
    A = repeat(A, 'h n -> b h n', b=B)
    B = repeat(B, 'h n -> b h n', b=B)
    C = repeat(C, 'h n -> b h n', b=B)
    
    # 计算离散化参数
    deltaA = torch.einsum('bhl,bhn->bhln', delta, A)
    deltaB = torch.einsum('bhl,bhn->bhln', delta, B)
    
    # 计算状态转移
    x = torch.zeros(B, H, N, device=u.device, dtype=u.dtype)
    us = []
    
    for t in range(L):
        # 更新状态
        x = x * torch.exp(deltaA[:, :, t, :])
        x = x + deltaB[:, :, t, :] * u[:, :, t].unsqueeze(-1)
        
        # 计算输出
        y = torch.einsum('bhn,bhn->bh', x, C)
        
        if D is not None:
            y = y + D * u[:, :, t]
        
        if z is not None:
            y = y + z[:, :, t]
            
        us.append(y)
    
    return torch.stack(us, dim=-1)  # (B, H, L) 