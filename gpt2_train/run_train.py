"""
@file   : run_train.py
@time   : 2026-01-03
"""
# 加载数据 -> 实现dataset -> Dataloader -> 模型 ->  优化器 -> 损失函数 -> 训练过程 -> 验证过程 -> inference
from config import set_args
from data_helper import load_data, MyDataset
from transformers import GPT2Tokenizer,GPT2LMHeadModel


if __name__ == '__main__':
    args = set_args()

    # 1. 加载数据
    train_data = load_data(args.train_data_path)
    test_data = load_data(args.test_data_path)
    print("训练集:", len(train_data))
    print("测试集:", len(test_data))

    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_pretrain)   # 实例化一个分词器

    # 2. 实现dataset
    train_dataset = MyDataset(train_data, tokenizer)
    print(train_dataset[32])




