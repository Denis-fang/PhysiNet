import torch
import torch.nn.functional as F

def selective_scan_fn(x, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
    """
    简化版的selective scan实现
    x: input tensor of shape (batch, dim, len)
    dt: delta tensor of shape (batch, dim, len)
    A, B, C: SSM parameters
    D: skip connection parameter
    z: optional input mixing
    """
    batch, dim, length = x.shape
    
    # 应用delta gate
    if delta_softplus:
        dt = F.softplus(dt + delta_bias) if delta_bias is not None else F.softplus(dt)
    
    # 计算状态更新
    u = torch.zeros(batch, dim, dtype=x.dtype, device=x.device)
    outputs = []
    
    for t in range(length):
        # 更新隐藏状态
        u = u * torch.exp(A * dt[..., t:t+1])  # 指数衰减
        u = u + B * x[..., t:t+1]  # 输入注入
        
        # 计算输出
        y = C * u  # 输出投影
        if D is not None:
            y = y + D * x[..., t:t+1]  # skip connection
        if z is not None:
            y = y + z[..., t:t+1]  # 输入混合
            
        outputs.append(y)
    
    return torch.cat(outputs, dim=-1) 