import os
from .base_loader import BaseDataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import glob
import logging

logger = logging.getLogger(__name__)

# 自定义CelebA数据集类
class CustomCelebaDataset(Dataset):
    def __init__(self, img_dir, split='train', transform=None, train_ratio=0.8, max_images=3000):
        """
        自定义CelebA数据集加载器
        
        参数:
            img_dir (str): 图像目录路径 (img_align_celeba)
            split (str): 'train'或'test'
            transform: 图像变换
            train_ratio: 训练集比例
            max_images: 最大图像数量限制
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # 获取所有图像文件
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        
        if not self.img_paths:
            raise RuntimeError(f"在 {img_dir} 中找不到图像文件")
        
        # 限制图像数量
        if max_images is not None:
            self.img_paths = self.img_paths[:max_images]
            
        logger.info(f"找到 {len(self.img_paths)} 张图像")
        
        # 划分训练集和测试集
        num_train = int(len(self.img_paths) * train_ratio)
        
        if split == 'train':
            self.img_paths = self.img_paths[:num_train]
        else:  # 'test'
            self.img_paths = self.img_paths[num_train:]
            
        logger.info(f"{split}集包含 {len(self.img_paths)} 张图像")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        try:
            # 使用高质量的图像加载方式
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            # 创建目标图像（无损坏的原始图像）
            target = image.clone()
            
            return image, target
        except Exception as e:
            logger.error(f"加载图像 {img_path} 时出错: {str(e)}")
            # 返回一个黑色图像作为替代
            if self.transform:
                # 创建一个空白图像并应用变换
                blank = Image.new('RGB', (256, 256), (0, 0, 0))
                blank_tensor = self.transform(blank)
                return blank_tensor, blank_tensor
            else:
                # 如果没有变换，返回零张量
                return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256)

class CelebALoader(BaseDataLoader):
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
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC),  # 使用双三次插值提高质量
            transforms.CenterCrop(self.image_size),  # 中心裁剪，确保人脸居中
        ])
        
        # 数据增强
        if config['enabled']:
            if config['random_flip']:
                transform_list.append(transforms.RandomHorizontalFlip())
            
            if config['color_jitter']['enabled']:
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=config['color_jitter']['brightness'],
                        contrast=config['color_jitter']['contrast'],
                        saturation=config['color_jitter']['saturation'],
                        hue=config['color_jitter']['hue']
                    )
                )
        
        # 转换为张量并归一化
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        
        return transforms.Compose(transform_list)
        
    def get_dataset(self):
        """获取CelebA数据集"""
        # 使用相对路径
        img_dir = os.path.join('data', 'celeba', 'img_align_celeba')
        
        # 检查目录是否存在
        if not os.path.exists(img_dir):
            logger.warning(f"未找到本地CelebA数据集目录: {img_dir}")
            raise FileNotFoundError(f"CelebA数据集目录不存在: {img_dir}")
        
        try:
            # 获取最大图像数量，默认为3000
            max_images = self.config['data'].get('max_images', 3000)
            
            train_dataset = CustomCelebaDataset(
                img_dir=img_dir,
                split='train',
                transform=self.transform,
                train_ratio=self.config['data'].get('train_val_split', 0.8),
                max_images=max_images
            )
            
            test_dataset = CustomCelebaDataset(
                img_dir=img_dir,
                split='test',
                transform=self.transform,
                train_ratio=self.config['data'].get('train_val_split', 0.8),
                max_images=max_images
            )
            
            logger.info(f"本地CelebA数据集加载成功，训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
            
            return {
                'train': train_dataset,
                'test': test_dataset
            }
        except Exception as e:
            logger.error(f"本地CelebA数据集加载失败: {str(e)}")
            raise
    
    def get_train_loader(self):
        """获取训练数据加载器"""
        return DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True  # 丢弃最后一个不完整的批次
        )
    
    def get_test_loader(self):
        """获取测试数据加载器"""
        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False  # 保留所有测试样本
        ) 