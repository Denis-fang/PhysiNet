import matplotlib.pyplot as plt
import json
import os

def plot_training_curves(log_file):
    """绘制训练曲线"""
    with open(log_file, 'r') as f:
        logs = [json.loads(line) for line in f]
    
    epochs = range(len(logs))
    losses = [log['loss'] for log in logs]
    psnrs = [log['psnr'] for log in logs]
    ssims = [log['ssim'] for log in logs]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.plot(epochs, losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(epochs, psnrs)
    ax2.set_title('PSNR')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    
    ax3.plot(epochs, ssims)
    ax3.set_title('SSIM')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('SSIM')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close() 