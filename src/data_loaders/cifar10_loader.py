from .base_loader import BaseDataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

class CIFAR10Loader(BaseDataLoader):
    def __init__(self, config, batch_size, image_size):
        self.batch_size = batch_size
        self.image_size = image_size
        super().__init__(config)
    
    def get_transforms(self):
        """获取数据转换"""
        config = self.config['augmentation']
        transform_list = []
        
        # 基础调整
        transform_list.extend([
            transforms.Resize((self.image_size, self.image_size)),  # 32x32 -> 64x64
            transforms.RandomHorizontalFlip(),  # 添加水平翻转
            transforms.RandomRotation(10),      # 添加随机旋转
            transforms.ColorJitter(            # 添加颜色抖动
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        
        return transforms.Compose(transform_list)
        
    def get_dataset(self):
        """获取数据集"""
        try:
            train_dataset = datasets.CIFAR10(
                root=self.config['data']['data_dir'],
                train=True,
                transform=self.transform,
                download=True
            )
            
            test_dataset = datasets.CIFAR10(
                root=self.config['data']['data_dir'],
                train=False,
                transform=self.transform,
                download=True
            )
            
            logger.info(f"CIFAR10数据集加载成功，训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
            
            return {
                'train': train_dataset,
                'test': test_dataset
            }
        except Exception as e:
            logger.error(f"CIFAR10数据集加载失败: {str(e)}")
            raise
    
    def get_train_loader(self):
        """获取训练数据加载器"""
        return DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def get_test_loader(self):
        """获取测试数据加载器"""
        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ) 