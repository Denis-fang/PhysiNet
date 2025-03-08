from .base_loader import BaseDataLoader
from torchvision import transforms
import torch
import os
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance
import random
import numpy as np
import PIL

class NuScenesDataset(Dataset):
    """nuScenes数据集"""
    def __init__(self, root_dir, transform=None, split='train', split_ratio=0.8, seed=42, max_images=3000):
        self.root_dir = root_dir
        self.transform = transform
        self.max_images = max_images
        
        # 获取所有图像文件
        self.image_files = []
        for file in os.listdir(root_dir):
            if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.jpeg') or file.endswith('.png'):
                self.image_files.append(os.path.join(root_dir, file))
        
        # 设置随机种子以确保可重复性
        random.seed(seed)
        
        # 打乱文件列表
        random.shuffle(self.image_files)
        
        # 限制最大图片数量
        if len(self.image_files) > self.max_images:
            print(f"限制数据集大小：从 {len(self.image_files)} 张图片减少到 {self.max_images} 张")
            self.image_files = self.image_files[:self.max_images]
        
        # 根据split_ratio分割训练集和测试集
        split_idx = int(len(self.image_files) * split_ratio)
        
        if split == 'train':
            self.image_files = self.image_files[:split_idx]
        else:  # 'test'
            self.image_files = self.image_files[split_idx:]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            # 返回图像和标签（标签为0，因为我们只关心图像重建）
            return image, 0
        except (PIL.UnidentifiedImageError, OSError, IOError) as e:
            # 记录错误
            print(f"警告: 无法加载图像 {img_path}，错误: {str(e)}")
            
            # 创建一个替代图像（黑色图像）
            if self.transform:
                # 创建一个与变换后图像大小相同的黑色图像
                dummy_img = torch.zeros(3, 256, 256)
            else:
                # 创建一个默认大小的黑色图像
                dummy_img = Image.new('RGB', (256, 256), color='black')
                if self.transform:
                    dummy_img = self.transform(dummy_img)
            
            return dummy_img, 0

class NuScenesLoader(BaseDataLoader):
    def __init__(self, config, batch_size, image_size):
        """初始化NuScenes数据加载器"""
        print(f"初始化NuScenesLoader，图像大小: {image_size}")
        self.batch_size = batch_size
        self.image_size = image_size
        self.config = config
        self.transform_train = self.get_train_transforms()
        self.transform_test = self.get_test_transforms()
        super().__init__(config)
        
        # 验证数据集
        if self.config.get('data', {}).get('validate_dataset', False):
            self.validate_dataset()
    
    def validate_dataset(self):
        """验证数据集中的图像是否可以正常加载"""
        print("正在验证数据集...")
        datasets = self.get_dataset()
        
        # 验证训练集
        train_dataset = datasets['train']
        invalid_train = self._validate_images(train_dataset, "训练集")
        
        # 验证测试集
        test_dataset = datasets['test']
        invalid_test = self._validate_images(test_dataset, "测试集")
        
        # 报告结果
        total_invalid = invalid_train + invalid_test
        if total_invalid > 0:
            print(f"警告: 数据集中发现 {total_invalid} 个无效图像")
        else:
            print("数据集验证完成，所有图像均有效")
    
    def _validate_images(self, dataset, name):
        """验证数据集中的图像"""
        invalid_count = 0
        for i in range(len(dataset)):
            try:
                # 尝试加载图像
                img_path = dataset.image_files[i]
                with Image.open(img_path) as img:
                    # 只需要检查图像是否可以打开
                    pass
            except Exception as e:
                invalid_count += 1
                print(f"错误: {name} 中的图像 {img_path} 无法加载: {str(e)}")
                
                # 如果发现太多无效图像，提前停止
                if invalid_count > 10:
                    print(f"发现过多无效图像，停止验证 {name}")
                    break
        
        return invalid_count
    
    def get_train_transforms(self):
        """获取训练数据转换，包含数据增强"""
        print(f"创建训练数据转换，图像大小: {self.image_size}")
        config = self.config.get('augmentation', {})
        transform_list = []
        
        # 基本变换
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        transform_list.append(transforms.RandomHorizontalFlip())
        
        # 条件变换
        if config.get('random_flip', True):
            transform_list.append(transforms.RandomVerticalFlip())
        
        if config.get('random_rotation', True):
            transform_list.append(transforms.RandomRotation(15))
        
        # 颜色增强，适合车辆图像
        if config.get('color_jitter', {}).get('enabled', True):
            transform_list.append(transforms.ColorJitter(
                brightness=config.get('color_jitter', {}).get('brightness', 0.1),
                contrast=config.get('color_jitter', {}).get('contrast', 0.1),
                saturation=config.get('color_jitter', {}).get('saturation', 0.1),
                hue=config.get('color_jitter', {}).get('hue', 0.05)
            ))
        
        # 最终变换
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(
            mean=[0.5, 0.5, 0.5],  # 使用中性均值
            std=[0.5, 0.5, 0.5]    # 使用中性标准差
        ))
        
        return transforms.Compose(transform_list)
    
    def get_test_transforms(self):
        """获取测试数据转换，不包含随机增强"""
        print(f"创建测试数据转换，图像大小: {self.image_size}")
        
        # 测试时只进行调整大小和标准化
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # 使用中性均值
                std=[0.5, 0.5, 0.5]    # 使用中性标准差
            )
        ]
        
        return transforms.Compose(transform_list)
        
    def get_transforms(self):
        """为兼容性保留，返回训练变换"""
        return self.get_train_transforms()
        
    def get_dataset(self):
        """获取数据集"""
        data_dir = self.config['data']['data_dir']
        root_dir = os.path.join(data_dir, 'nuimages-v1.0-mini', 'CMA')
        
        # 获取最大图片数量参数
        max_images = self.config.get('data', {}).get('max_images', 3000)
        
        # 创建训练集和测试集
        train_dataset = NuScenesDataset(
            root_dir=root_dir,
            transform=self.transform_train,
            split='train',
            max_images=max_images
        )
        
        test_dataset = NuScenesDataset(
            root_dir=root_dir,
            transform=self.transform_test,
            split='test',
            max_images=max_images
        )
        
        return {'train': train_dataset, 'test': test_dataset} 