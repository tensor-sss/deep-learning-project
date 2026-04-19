"""
@file   : config.py
@time   : 2026-04-19
"""

import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='./data/train_data.csv', help='训练集')
    parser.add_argument('--valid_data_path', type=str, default='./data/valid_data.csv', help='验证集')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=5, help='训练轮次')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--pretrain_model', type=str, default='./bert_pretrain', help='预训练模型')
    return parser.parse_args()



