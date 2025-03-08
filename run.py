import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='PhysiNet Training')
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--lr', type=float, help='learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer()
    
    # 如果命令行指定了参数，则覆盖配置文件中的设置
    if args.epochs:
        trainer.config['training']['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['training']['batch_size'] = args.batch_size
    if args.lr:
        trainer.config['training']['learning_rate'] = args.lr
    
    trainer.train()