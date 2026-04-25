"""
@file   : config_tower.py
@time   : 2026-04-25
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='./data/train_tower.jsonl', help='训练集(jsonl)')
    parser.add_argument('--valid_data_path', type=str, default='./data/valid_tower.jsonl', help='验证集(jsonl)')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=5, help='训练轮次')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--max_len', type=int, default=64, help='最大长度')
    parser.add_argument('--temperature', type=float, default=0.05, help='对比学习温度系数')
    parser.add_argument('--pretrain_model', type=str, default='./bert_pretrain', help='预训练模型')
    parser.add_argument('--save_model_path', type=str, default='./best_tower_model.pt', help='最优模型保存路径')
    return parser.parse_args()
