from .mamba_physinet import MambaPhysiNet
from .seponet import SepONet
from .pde_net import PDENet
from .mixture_of_mamba import MixtureOfMamba
from .gcdd_layer import GCDDLayer

def get_model(config):
    """根据配置创建模型"""
    model_type = config['model']['type']
    
    if model_type == 'mamba':
        return MambaPhysiNet(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            d_model=config['model'].get('d_model', 64),
            d_state=config['model'].get('d_state', 16),
            d_conv=config['model'].get('d_conv', 4),
            expand=config['model'].get('expand', 2),
            config=config  # 传入完整配置
        )
    elif model_type == 'seponet':
        # SepONet只接受一个config参数
        return SepONet(config=config)
    elif model_type == 'gcdd':
        # 创建基础配置字典
        base_config = {
            'in_channels': config['model']['in_channels'],
            'out_channels': config['model']['out_channels'],
            'd_model': config['model'].get('d_model', 64)
        }
        return PDENet(
            **base_config,
            time_steps=config['model']['gcdd']['time_steps'],
            diffusion_coeff=config['model']['gcdd']['diffusion_coeff']
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}") 