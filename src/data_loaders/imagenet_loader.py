from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from .base_loader import BaseDataLoader

class ImageNetDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, num_workers=4):
        super().__init__(data_dir, batch_size, num_workers)
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def get_train_loader(self):
        train_dir = os.path.join(self.data_dir, 'train')
        train_dataset = datasets.ImageFolder(
            train_dir,
            transform=self.transform
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
    def get_test_loader(self):
        val_dir = os.path.join(self.data_dir, 'val')
        test_dataset = datasets.ImageFolder(
            val_dir,
            transform=self.transform
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        ) 