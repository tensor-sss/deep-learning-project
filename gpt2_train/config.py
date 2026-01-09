"""
@file   : config.py
@time   : 2026-01-03
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='./data/train_data.jsonl')
    parser.add_argument('--test_data_path', type=str, default='./data/test_data.jsonl')
    parser.add_argument('--gpt2_pretrain', type=str, default='./gpt2_pretrain')
    return parser.parse_args()


