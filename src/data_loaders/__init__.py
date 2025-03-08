from torch.utils.data import DataLoader
import logging

# 导入所有数据加载器
from .cifar10_loader import CIFAR10Loader
from .imagenet_loader import ImageNetDataLoader
from .flowers102_loader import Flowers102Loader
from .celeba_loader import CelebALoader
from .brast2021_loader import BRAST2021Loader
from .nuscenes_loader import NuScenesLoader

logger = logging.getLogger(__name__)

def get_dataloader(dataset_name, data_dir, batch_size, num_workers=4):
    """获取数据加载器（旧接口，保留兼容性）"""
    # 保留旧接口以兼容现有代码
    pass

def get_dataloaders(config):
    """
    根据配置获取数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
    dataset_name = config['data']['dataset'].lower()
    batch_size = config['training']['batch_size']
    image_size = config['data']['image_size']
    
    logger.info(f"初始化数据加载器: {dataset_name}, 批量大小: {batch_size}, 图像大小: {image_size}")
    
    if dataset_name == 'cifar10':
        data_loader = CIFAR10Loader(
            config=config,
            batch_size=batch_size,
            image_size=image_size
        )
    elif dataset_name == 'celeba':
        data_loader = CelebALoader(
            config=config,
            batch_size=batch_size,
            image_size=image_size
        )
    elif dataset_name == 'flowers102':
        data_loader = Flowers102Loader(
            config=config,
            batch_size=batch_size,
            image_size=image_size
        )
    elif dataset_name == 'brast2021':
        data_loader = BRAST2021Loader(
            config=config,
            batch_size=batch_size,
            image_size=image_size
        )
    elif dataset_name == 'nuscenes':
        data_loader = NuScenesLoader(
            config=config,
            batch_size=batch_size,
            image_size=image_size
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return data_loader.get_train_loader(), data_loader.get_test_loader() 