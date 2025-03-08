import torch
import torch.nn.functional as F

def selective_scan_fn(
    u: torch.Tensor,  # shape: (B, D, L)
    delta: torch.Tensor,  # shape: (D)
    A: torch.Tensor,  # shape: (D, N)
    B: torch.Tensor,  # shape: (1, D, N)
    C: torch.Tensor,  # shape: (1, D, N)
    D: torch.Tensor = None,  # shape: (D)
    z: torch.Tensor = None,  # shape: (B, D, L)
    delta_bias: torch.Tensor = None,
    delta_softplus: bool = True,
    return_last_state: bool = False
):
    """
    完整的selective scan实现
    参数维度说明：
    B: batch_size
    D: d_inner
    L: sequence_length
    N: d_state
    """
    batch_size, dim, length = u.shape
    n_state = A.shape[1]
    
    # 处理delta
    if delta_softplus:
        delta = F.softplus(delta)
        if delta_bias is not None:
            delta = delta + delta_bias
    
    # 初始化状态
    x = torch.zeros(batch_size, dim, n_state, device=u.device, dtype=u.dtype)
    ys = []
    
    # 扩展维度
    A = A.unsqueeze(0).expand(batch_size, -1, -1)  # [B, D, N]
    delta = delta.unsqueeze(-1).unsqueeze(0).expand(batch_size, -1, 1)  # [B, D, 1]
    B = B.expand(batch_size, -1, -1)  # [B, D, N]
    C = C.expand(batch_size, -1, -1)  # [B, D, N]
    
    # 主循环
    for t in range(length):
        # 更新状态
        dA = torch.exp(A * delta)  # [B, D, N]
        x = x * dA  # [B, D, N]
        
        # 更新状态
        u_t = u[:, :, t:t+1]  # [B, D, 1]
        x = x + u_t * B  # [B, D, N]
        
        # 计算输出
        y = (x * C).sum(dim=-1)  # [B, D]
        
        # 添加skip connection
        if D is not None:
            y = y + D * u[:, :, t]
        
        # 添加输入混合
        if z is not None:
            y = y + z[:, :, t]
        
        ys.append(y)
    
    # 组装输出
    y = torch.stack(ys, dim=-1)  # [B, D, L]
    
    if return_last_state:
        return y, x
    return y 