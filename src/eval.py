import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.utils import calculate_psnr, calculate_ssim

class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for data, _ in tqdm(self.test_loader, desc='Evaluating'):
                data = data.to(self.device)
                output = self.model(data)
                
                # 计算PSNR和SSIM
                psnr = calculate_psnr(output, data)
                ssim = calculate_ssim(output, data)
                
                total_psnr += psnr
                total_ssim += ssim
                
        avg_psnr = total_psnr / len(self.test_loader)
        avg_ssim = total_ssim / len(self.test_loader)
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim
        } 