import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .gcdd_layer import GCDDLayer
from .mamba_block import MambaBlock, SelectiveScan

class SEModule(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SeparableConv2d(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        # 确保channels能被num_heads整除
        assert channels % num_heads == 0, f"通道数 {channels} 必须能被注意力头数 {num_heads} 整除"
        self.head_dim = channels // num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        # 添加层归一化增加稳定性
        self.norm = nn.LayerNorm([channels, 1, 1])
        # 初始化参数
        self._reset_parameters()
        
    def _reset_parameters(self):
        # 使用更稳定的初始化方法
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 添加数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"警告: MultiHeadAttention输入包含NaN或Inf值")
            # 替换NaN和Inf值
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 应用层归一化
        x_norm = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x_norm = torch.nn.functional.layer_norm(x_norm, [C])
        x_norm = x_norm.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # 生成QKV
        qkv = self.qkv(x_norm).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)
        
        # 计算注意力分数，添加数值稳定性
        scale = float(self.head_dim) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # 裁剪极端值
        attn = torch.clamp(attn, min=-50.0, max=50.0)
        
        # 应用softmax
        attn = attn.softmax(dim=-1)
        
        # 防止梯度爆炸
        attn = torch.nan_to_num(attn, nan=1.0/attn.shape[-1])
        
        # 计算输出
        x = (attn @ v).reshape(B, C, H, W)
        x = self.proj(x)
        
        # 最后再次检查输出
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"警告: MultiHeadAttention输出包含NaN或Inf值")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return x

class MultiScaleAttention(nn.Module):
    """多尺度注意力模块"""
    def __init__(self, channels, num_heads=8, scales=[1, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.attention_heads = nn.ModuleList([
            MultiHeadAttention(channels, num_heads=num_heads)
            for _ in scales
        ])
        self.fusion = nn.Conv2d(channels * len(scales), channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        multi_scale_features = []
        
        for scale, attn in zip(self.scales, self.attention_heads):
            # 调整特征图尺寸
            if scale != 1:
                h, w = int(H * scale), int(W * scale)
                scaled_x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            else:
                scaled_x = x
                
            # 应用注意力
            attended = attn(scaled_x)
            
            # 恢复原始尺寸
            if scale != 1:
                attended = F.interpolate(attended, size=(H, W), mode='bilinear', align_corners=True)
                
            multi_scale_features.append(attended)
            
        # 融合多尺度特征
        concat_features = torch.cat(multi_scale_features, dim=1)
        return self.fusion(concat_features)

class SepOBlock(nn.Module):
    """增强的SepONet基本块"""
    def __init__(self, channels, expansion=4, num_heads=8):
        super().__init__()
        hidden_channels = channels * expansion
        
        # 深度可分离卷积
        self.conv1 = SeparableConv2d(channels, hidden_channels)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        
        # 多尺度注意力
        self.multi_scale_attention = MultiScaleAttention(hidden_channels, num_heads)
        
        # 通道注意力
        self.se = SEModule(hidden_channels)
        
        # 第二个深度可分离卷积
        self.conv2 = SeparableConv2d(hidden_channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.GELU()
        
        # 特征重标定
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        identity = x
        
        # 第一阶段处理
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        # 多尺度注意力
        out = self.multi_scale_attention(out)
        
        # 通道注意力
        out = self.se(out)
        
        # 第二阶段处理
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 特征重标定
        out = out * self.scale + self.bias
        
        # 残差连接
        out += identity
        out = self.act2(out)
        
        return out

class MixOfMambaModule(nn.Module):
    """Mixture of Mamba模块"""
    def __init__(self, channels, num_experts=3, d_state=64, d_conv=4):
        super().__init__()
        self.num_experts = num_experts
        
        # 专家网络 - 使用Mamba块
        self.experts = nn.ModuleList([
            MambaBlock(channels, d_state=d_state, d_conv=d_conv)
            for _ in range(num_experts)
        ])
        
        # 路由网络 - 修复LayerNorm尺寸问题
        self.conv_dim = 64
        self.router = nn.Sequential(
            nn.Conv2d(channels, self.conv_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.conv_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # 专家融合
        self.fusion = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 保存输入形状
        input_shape = x.shape
        seq_len = H * W
        
        # 将2D特征转换为序列
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 计算路由权重
        x_router = x.clone()  # 使用副本进行路由计算
        weights = self.router(x_router)  # [B, num_experts]
        
        # 专家输出
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            try:
                # 通过Mamba块处理序列
                out_seq = expert(x_seq)  # [B, H*W, C]
                
                # 确保输出序列长度正确
                if out_seq.size(1) != seq_len:
                    # 如果长度不匹配，使用插值调整长度
                    out_seq_reshaped = out_seq.transpose(1, 2)  # [B, C, seq_len']
                    out_seq_resized = F.interpolate(
                        out_seq_reshaped.unsqueeze(3),  # [B, C, seq_len', 1]
                        size=(seq_len, 1),
                        mode='nearest'
                    ).squeeze(3)  # [B, C, seq_len]
                    out_seq = out_seq_resized.transpose(1, 2)  # [B, seq_len, C]
                
                # 转回2D特征
                out = out_seq.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
                expert_outputs.append(out)
            except RuntimeError as e:
                print(f"专家{i}处理失败: {str(e)}")
                # 如果处理失败，使用输入作为输出
                expert_outputs.append(x)
            
        # 堆叠专家输出
        stacked_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, C, H, W]
        
        # 应用路由权重
        weights = weights.view(B, self.num_experts, 1, 1, 1)
        weighted_sum = (stacked_outputs * weights).sum(dim=1)  # [B, C, H, W]
        
        # 最终融合
        output = self.fusion(weighted_sum)
        
        return output

class SepONet(nn.Module):
    """增强版SepONet，集成了物理约束（高斯曲率驱动扩散）和MixOfMamba"""
    def __init__(self, config):
        super().__init__()
        # SepONet特定参数
        self.num_blocks = config['model']['seponet']['num_blocks']  # 4
        self.expansion = config['model']['seponet']['expansion']    # 2
        self.attention_heads = config['model']['seponet']['attention_heads']  # 4
        self.d_model = config['model']['d_model']  # 从config中获取d_model
        
        # 输入处理
        self.in_channels = config['model']['in_channels']
        self.out_channels = config['model']['out_channels']
        
        # 输入卷积层
        self.input_conv = nn.Conv2d(self.in_channels, self.d_model, kernel_size=3, padding=1)
        self.input_norm = nn.BatchNorm2d(self.d_model)
        self.input_act = nn.GELU()
        
        # 可分离算子块
        self.sep_blocks = nn.ModuleList([
            SeparableOperatorBlock(
                channels=self.d_model,
                expansion=self.expansion,
                num_heads=self.attention_heads
            ) for _ in range(self.num_blocks)
        ])
        
        # 添加GCDD层作为物理约束
        gcdd_config = config['model']['gcdd']
        self.use_gcdd = config['model'].get('use_gcdd', True)  # 默认启用GCDD
        self.gcdd = GCDDLayer(
            time_steps=gcdd_config.get('time_steps', 10),
            dt=gcdd_config.get('dt', 0.01),
            alpha=gcdd_config.get('alpha', 0.1),
            beta=gcdd_config.get('beta', 0.01)
        )
        
        # GCDD损失权重
        self.gcdd_weight = config['model'].get('gcdd_weight', 0.1)
        
        # 添加MixOfMamba模块
        self.use_mamba = config['model'].get('use_mamba', True)  # 默认启用MixOfMamba
        if self.use_mamba:
            mamba_config = config['model'].get('mamba', {})
            self.mamba_module = MixOfMambaModule(
                channels=self.d_model,
                num_experts=mamba_config.get('num_experts', 3),
                d_state=mamba_config.get('d_state', 64),
                d_conv=mamba_config.get('d_conv', 4)
            )
            self.diversity_weight = mamba_config.get('diversity_weight', 0.01)
        
        # 输出卷积层
        self.output_conv = nn.Conv2d(self.d_model, self.out_channels, kernel_size=3, padding=1)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        # 初始化输入卷积
        nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.input_conv.bias is not None:
            nn.init.zeros_(self.input_conv.bias)
            
        # 初始化输出卷积
        nn.init.kaiming_normal_(self.output_conv.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.output_conv.bias)
        
    def forward(self, x):
        """前向传播"""
        # 输入处理
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = self.input_act(x)
        
        # 检查数值稳定性
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"警告: SepONet输入处理后包含NaN或Inf值")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 通过所有可分离算子块
        for i, block in enumerate(self.sep_blocks):
            x_prev = x  # 保存前一层的输出
            x = block(x)
            
            # 检查每个块的输出是否有NaN
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"警告: 第{i+1}个SeparableOperatorBlock输出包含NaN或Inf值")
                # 如果出现NaN，使用前一层的输出
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                # 如果整个张量都是NaN，回退到前一层的输出
                if torch.isnan(x).all() or torch.isinf(x).all():
                    print(f"严重警告: 第{i+1}个块输出全是NaN或Inf，回退到前一层输出")
                    x = x_prev
        
        # 应用MixOfMamba (如果启用)
        if self.use_mamba:
            x_before_mamba = x  # 保存MixOfMamba前的特征
            x = self.mamba_module(x)
            
            # 检查MixOfMamba输出是否有NaN
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"警告: MixOfMamba输出包含NaN或Inf值")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                # 如果整个张量都是NaN，回退到MixOfMamba前的特征
                if torch.isnan(x).all() or torch.isinf(x).all():
                    print(f"严重警告: MixOfMamba输出全是NaN或Inf，跳过MixOfMamba处理")
                    x = x_before_mamba
        
        # 应用物理约束 (GCDD)
        if self.use_gcdd:
            x_before_gcdd = x  # 保存GCDD前的特征
            x = self.gcdd(x)
            
            # 检查GCDD输出是否有NaN
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"警告: GCDD输出包含NaN或Inf值")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                # 如果整个张量都是NaN，回退到GCDD前的特征
                if torch.isnan(x).all() or torch.isinf(x).all():
                    print(f"严重警告: GCDD输出全是NaN或Inf，跳过GCDD处理")
                    x = x_before_gcdd
            
        # 输出处理
        x = self.output_conv(x)
        
        # 最终检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"警告: SepONet最终输出包含NaN或Inf值")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x
    
    def compute_gcdd_loss(self, pred, target):
        """计算GCDD相关的损失"""
        total_loss = 0.0
        
        # GCDD损失
        if self.use_gcdd:
            gcdd_loss = self.gcdd.loss_function(pred, target) * self.gcdd_weight
            total_loss += gcdd_loss
            
        # MixOfMamba多样性损失
        if self.use_mamba:
            mamba_loss = self.compute_mamba_diversity_loss() * self.diversity_weight
            total_loss += mamba_loss
            
        return total_loss
        
    def compute_mamba_diversity_loss(self):
        """计算MixOfMamba的多样性损失，鼓励不同专家学习不同的特征"""
        if not self.use_mamba:
            return 0.0
            
        # 获取专家权重
        expert_weights = []
        for name, param in self.mamba_module.named_parameters():
            if 'experts' in name and 'weight' in name:  # 只使用权重参数
                # 将参数展平为一维向量
                flat_param = param.view(-1)
                # 如果参数太大，采样一部分
                if flat_param.numel() > 10000:
                    indices = torch.randperm(flat_param.numel())[:10000]
                    flat_param = flat_param[indices]
                expert_weights.append(flat_param)
                
        # 如果没有专家权重，返回0
        if len(expert_weights) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        # 计算专家权重之间的相似度
        similarity = 0.0
        num_pairs = 0
        
        for i in range(len(expert_weights)):
            for j in range(i+1, len(expert_weights)):
                # 确保两个向量长度相同
                min_len = min(expert_weights[i].size(0), expert_weights[j].size(0))
                vec1 = expert_weights[i][:min_len]
                vec2 = expert_weights[j][:min_len]
                
                # 计算余弦相似度
                try:
                    cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1)
                    similarity += cos_sim.abs().mean()
                    num_pairs += 1
                except RuntimeError as e:
                    print(f"计算余弦相似度时出错: {str(e)}")
                    continue
                
        # 平均相似度作为多样性损失
        if num_pairs > 0:
            diversity_loss = similarity / num_pairs
        else:
            diversity_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            
        return diversity_loss

class SeparableOperatorBlock(nn.Module):
    def __init__(self, channels, expansion, num_heads):
        super().__init__()
        
        # 空间分离算子
        self.spatial_op = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels * expansion, 1)
        )
        
        # 通道分离算子
        self.channel_op = nn.Sequential(
            nn.Conv2d(channels * expansion, channels, 1),
            nn.GELU()
        )
        
        # 自注意力机制
        self.attention = MultiHeadAttention(channels, num_heads)
        
        # 添加层归一化增加稳定性
        self.norm1 = nn.LayerNorm([channels, 1, 1])
        self.norm2 = nn.LayerNorm([channels, 1, 1])
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        # 初始化空间分离算子
        for m in self.spatial_op.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # 初始化通道分离算子
        for m in self.channel_op.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """前向传播"""
        # 保存输入作为残差连接
        identity = x
        
        # 检查输入数值
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"警告: SeparableOperatorBlock输入包含NaN或Inf值")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 应用层归一化
        x_norm = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x_norm = torch.nn.functional.layer_norm(x_norm, x_norm.shape[-1:])
        x_norm = x_norm.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        # 空间分离算子
        spatial_features = self.spatial_op(x_norm)
        
        # 检查空间特征
        if torch.isnan(spatial_features).any() or torch.isinf(spatial_features).any():
            print(f"警告: 空间分离算子输出包含NaN或Inf值")
            spatial_features = torch.nan_to_num(spatial_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 通道分离算子
        channel_features = self.channel_op(spatial_features)
        
        # 检查通道特征
        if torch.isnan(channel_features).any() or torch.isinf(channel_features).any():
            print(f"警告: 通道分离算子输出包含NaN或Inf值")
            channel_features = torch.nan_to_num(channel_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 自注意力
        attention_features = self.attention(x)
        
        # 检查注意力特征
        if torch.isnan(attention_features).any() or torch.isinf(attention_features).any():
            print(f"警告: 自注意力输出包含NaN或Inf值")
            attention_features = torch.nan_to_num(attention_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 融合特征（使用缩放因子增加稳定性）
        output = channel_features * 0.5 + attention_features * 0.3 + identity * 0.2
        
        # 最终检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"警告: SeparableOperatorBlock最终输出包含NaN或Inf值")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            # 如果仍然有问题，回退到输入
            if torch.isnan(output).all() or torch.isinf(output).all():
                print(f"严重警告: 块输出全是NaN或Inf，回退到输入")
                output = identity
        
        return output

class PyramidPooling(nn.Module):
    """金字塔池化模块"""
    def __init__(self, channels):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )
            for size in [1, 2, 3, 6]
        ])
        
    def forward(self, x):
        features = [x]
        h, w = x.shape[2:]
        
        for pool in self.pools:
            feat = pool(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
            features.append(feat)
            
        return torch.cat(features, dim=1) 