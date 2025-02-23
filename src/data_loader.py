import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # 获取图像文件列表
        self.image_files = []
        data_path = os.path.join(data_dir, mode)
        for file in os.listdir(data_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                self.image_files.append(os.path.join(data_path, file))
                
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 这里可以添加数据预处理或数据增强的步骤
        return image, image  # 返回原图和目标图像对