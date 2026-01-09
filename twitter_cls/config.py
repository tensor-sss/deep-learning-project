"""
@file   : config.py
@time   : 2025-12-21
"""

import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default='./data/train_data.csv', help='训练集')
    parser.add_argument("--test_data_path", type=str, default='./data/dev_data.csv', help='测试集')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')

    parser.add_argument("--pretrain_model", type=str, default='./bert_en_pretrain', help='预训练模型的位置')

    # 从头开始训练自己的模型  0.01  ~  0.0001
    # 微调模型时候 学习率 不要太大   0.00001左右
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')

    # 微调模型  这个轮次 不需要太大
    parser.add_argument('--num_epochs', type=int, default=5, help='训练的轮数')
    return parser.parse_args()