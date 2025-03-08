import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import os
from torchvision.datasets import CelebA
from PIL import Image
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from src.data_loaders import get_dataloaders as get_dataloaders_new

class CelebADataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 获取所有jpg文件路径
        img_dir = os.path.join(data_dir, 'img_align_celeba')
        if not os.path.exists(img_dir):
            raise ValueError(f"图片目录不存在: {img_dir}")
            
        self.image_paths = glob(os.path.join(img_dir, '*.jpg'))
        if len(self.image_paths) == 0:
            raise ValueError(f"在 {img_dir} 中没有找到任何jpg图片")
            
        print(f"找到 {len(self.image_paths)} 张图片")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # 返回0作为标签占位符

def get_dataloaders(config):
    """获取数据加载器"""
    return get_dataloaders_new(config)

transform = transforms.Compose([
    transforms.Resize(80),  # 先调整到更大尺寸
    transforms.RandomCrop(64),  # 随机裁剪到目标尺寸
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # 增加随机旋转
    transforms.ColorJitter(
        brightness=0.2, 
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 添加随机平移
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])