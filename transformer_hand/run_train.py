"""
@file   : run_train.py
@time   : 2026-01-11
"""
# 加载数据 -> Dataset -> DataLoader -> 模型 -> 损失函数 -> 优化器 -> 写训练过程 -> 写验证过程(保存模型) -> 推理脚本
import json
from config import set_args
from data_helper import load_data, MyDataset, collate_fn
from torch.utils.data import DataLoader
from model import Transformer


if __name__ == '__main__':
    args = set_args()

    # 1. 加载数据
    train_data = load_data(args.train_data_path)
    test_data = load_data(args.test_data_path)
    print('数据样例:', train_data[0])
    print("训练集数量:", len(train_data))
    print("测试集数量:", len(test_data))
    # [{'question': xxx, "answer": xx}, {}, {}]
    vocab2id = json.load(open(args.vocab2id_path, 'r', encoding='utf8'))
    print("词表大小:", len(vocab2id))

    # 2. 实现Dataset  DataLoader
    train_dataset = MyDataset(train_data, vocab2id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = Transformer(len(vocab2id))
    for batch in train_dataloader:
        encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask = batch
        # print(encoder_input_ids.size(), encoder_attention_mask.size())
        # print(decoder_input_ids.size(), decoder_attention_mask.size())
        out = model(encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask)



















