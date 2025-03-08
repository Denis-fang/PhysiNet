from .base_loader import BaseDataLoader
from torchvision import datasets, transforms
import torch
from PIL import ImageFilter, ImageEnhance

class StructurePreservingTransform:
    """结构保持型变换，增强边缘和结构信息"""
    def __init__(self, sharpness_factor=0.3):
        self.sharpness_factor = sharpness_factor
        
    def __call__(self, img):
        # 增强锐度，保持边缘结构
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.0 + self.sharpness_factor)
        return img

class Flowers102Loader(BaseDataLoader):
    def __init__(self, config, batch_size, image_size):
        print(f"Initializing Flowers102Loader with image_size: {image_size}")  # 调试信息
        self.batch_size = batch_size
        self.image_size = image_size
        self.config = config
        self.transform_train = self.get_train_transforms()
        self.transform_test = self.get_test_transforms()
        super().__init__(config)  # 先调用父类的初始化
    
    def get_train_transforms(self):
        """获取训练数据转换，包含数据增强"""
        print(f"Creating train transforms with image_size: {self.image_size}")  # 调试信息
        config = self.config.get('augmentation', {})
        transform_list = []
        
        # 增强数据增强
        transform_list.extend([
            transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip() if config.get('random_flip', True) else transforms.Lambda(lambda x: x),  # 添加垂直翻转
            transforms.RandomRotation(30) if config.get('random_rotation', True) else transforms.Lambda(lambda x: x),    # 增加旋转角度
            transforms.ColorJitter(
                brightness=config.get('color_jitter', {}).get('brightness', 0.2),
                contrast=config.get('color_jitter', {}).get('contrast', 0.2),
                saturation=config.get('color_jitter', {}).get('saturation', 0.2),
                hue=config.get('color_jitter', {}).get('hue', 0.1)                       # 添加色调变化
            ) if config.get('color_jitter', {}).get('enabled', True) else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet均值
                std=[0.229, 0.224, 0.225]     # ImageNet标准差
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def get_test_transforms(self):
        """获取测试数据转换，不包含随机增强"""
        print(f"Creating test transforms with image_size: {self.image_size}")  # 调试信息
        
        # 测试时只进行调整大小和标准化，不进行随机变换
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),  # 使用中心裁剪而非随机裁剪
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet均值
                std=[0.229, 0.224, 0.225]     # ImageNet标准差
            )
        ]
        
        return transforms.Compose(transform_list)
        
    def get_transforms(self):
        """为兼容性保留，返回训练变换"""
        return self.get_train_transforms()
        
    def get_dataset(self):
        """获取数据集，为训练和测试使用不同的变换"""
        return {
            'train': datasets.Flowers102(
                root=self.config['data']['data_path'],
                split='train',
                transform=self.transform_train,
                download=True
            ),
            'test': datasets.Flowers102(
                root=self.config['data']['data_path'],
                split='test',
                transform=self.transform_test,
                download=True
            )
        } 