"""
@file   : config.py
@time   : 2026-03-01
"""


import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='./data/train_data_new.json', help='训练集')
    parser.add_argument('--dev_data_path', type=str, default='./data/dev_data_new.json', help='验证集')
    parser.add_argument('--label2id_path', type=str, default='./data/label2id.json', help='标签映射表')
    parser.add_argument('--bert_pretrain', type=str, default='./bert_pretrain', help='预训练模型')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--output_dir', type = str, default = './output', help = '输出路径')

    return parser.parse_args()
