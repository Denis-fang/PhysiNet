import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os

def get_dataloaders(dataset_name, batch_size=32, data_dir='data'):
    """获取数据加载器"""
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                       download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                      transform=transform)
                                      
    elif dataset_name.lower() == 'imagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_dir = os.path.join(data_dir, 'imagenet/train')
        val_dir = os.path.join(data_dir, 'imagenet/val')
        
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        test_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4)
    
    return train_loader, test_loader