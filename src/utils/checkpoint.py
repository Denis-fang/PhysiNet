import os
import torch
import glob

class CheckpointManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.best_psnr = float('-inf')  # 初始化为负无穷，确保第一个模型会被保存
        
    def save(self, model, epoch, loss, psnr, ssim):
        """保存检查点"""
        # 打印调试信息
        print(f"正在保存检查点: epoch={epoch}, psnr={psnr}, best_psnr={self.best_psnr}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': {
                'loss': loss,
                'psnr': psnr,
                'ssim': ssim
            }
        }
        
        # 保存当前检查点
        checkpoint_path = os.path.join(
            self.save_dir, 
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"已保存检查点到: {checkpoint_path}")
        
        # 保存最佳模型
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            best_model_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f'发现更好的模型！保存最佳模型，PSNR: {psnr:.2f}')
    
    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth'):
        """保存检查点（兼容train.py中的调用）"""
        # 保存当前检查点
        checkpoint_path = os.path.join(self.save_dir, filename)
        torch.save(state, checkpoint_path)
        print(f"已保存检查点到: {checkpoint_path}")
        
        # 如果是最佳模型，也保存为best_model.pth
        if is_best:
            best_model_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(state, best_model_path)
            print(f'发现更好的模型！保存最佳模型')
            
            # 更新best_psnr（如果state中包含psnr信息）
            if 'best_metrics' in state and 'psnr' in state['best_metrics']:
                self.best_psnr = state['best_metrics']['psnr']
                print(f"更新最佳PSNR: {self.best_psnr}")
            
    def _is_best(self, current_psnr):
        """检查是否是最佳模型"""
        return current_psnr > self.best_psnr
        
    def load_latest(self):
        """加载最新的检查点"""
        checkpoints = glob.glob(os.path.join(self.save_dir, 'checkpoint_*.pth'))
        if not checkpoints:
            print("没有找到任何检查点")
            return None
            
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"加载最新检查点: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        
        # 更新best_psnr
        if 'metrics' in checkpoint and 'psnr' in checkpoint['metrics']:
            psnr = checkpoint['metrics']['psnr']
            self.best_psnr = max(self.best_psnr, psnr)
            print(f"更新最佳PSNR: {self.best_psnr}")
        
        return checkpoint
        
    def load_best(self):
        """加载最佳模型"""
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f"加载最佳模型: {best_model_path}")
            checkpoint = torch.load(best_model_path)
            
            # 更新best_psnr
            if 'metrics' in checkpoint and 'psnr' in checkpoint['metrics']:
                self.best_psnr = checkpoint['metrics']['psnr']
                print(f"最佳模型PSNR: {self.best_psnr}")
            
            return checkpoint
        print("没有找到最佳模型")
        return None 