from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import abc

class BaseDataLoader(abc.ABC):
    """数据加载器基类"""
    def __init__(self, config):
        self.config = config
        self.transform = self.get_transforms()
        self.datasets = self.get_dataset()
        
    @abc.abstractmethod
    def get_transforms(self):
        """获取数据转换"""
        pass
        
    @abc.abstractmethod
    def get_dataset(self):
        """获取数据集"""
        pass
        
    def get_train_loader(self):
        return DataLoader(
            self.datasets['train'],
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
    def get_test_loader(self):
        return DataLoader(
            self.datasets['test'],
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ) 