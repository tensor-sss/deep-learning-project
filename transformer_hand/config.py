"""
@file   : config.py
@time   : 2026-01-11
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--train_data_path', type=str, default='./data/train_data.jsonl')
    parser.add_argument('--test_data_path', type=str, default='./data/test_data.jsonl')
    parser.add_argument('--vocab2id_path', type=str, default='./data/vocab2id.json')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=3e-3)

    return parser.parse_args()
