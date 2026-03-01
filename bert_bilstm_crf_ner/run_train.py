"""
@file   : run_train.py
@time   : 2026-03-01
"""

# config.py -> 加载数据 -> dataset -> dataloader -> 模型 -> 优化器 损失函数 -> 训练过程 -> 验证过程 -> 训练结束后 写推理过程inference.py
import json
from config import set_args
from data_helper import load_data, NERDataset, collate_fn
from transformers.models.bert import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from model import Model
from torch import nn
from torch.optim import AdamW


def evaluate():
    # 逐token准确率 召回率  。。。。
    pass


if __name__ == '__main__':
    args = set_args()

    # 1. 加载数据
    train_data = load_data(args.train_data_path)
    dev_data = load_data(args.dev_data_path)
    # print("训练集样本数:", len(train_data))    # 训练集样本数: 3821
    # print("验证集样本数:", len(dev_data))   # 验证集样本数: 463
    # print(train_data[32])
    # print(dev_data[24])

    label2id = json.load(open(args.label2id_path, 'r', encoding='utf8'))
    label_num = len(label2id)


    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain)

    # 2. 实现dataset    文本 和标签 转成对应id
    train_dataset = NERDataset(train_data, tokenizer, label2id)

    # 3. 实现dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)  # 封装batch 转tensor

    model = Model(label_num)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        for batch in train_dataloader:
            input_ids, attention_mask, label_ids = batch  # (input_ids, attention_mask, label_ids)
            # print(input_ids.size())  # torch.Size([2, 37])
            # print(attention_mask.size())   #
            # print(label_ids.size())

            logits = model(input_ids, attention_mask)   # model.forward(input_ids, attention_mask)
            # print(logits.size())   # torch.Size([batch_size, max_len, 17])

            logits = logits.view(-1, label_num)
            label_ids = label_ids.view(-1)
            # print(logits.size())
            # print(label_ids.size())
            loss = loss_func(logits, label_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        evaluate()  #xxx
        # 每一轮保存一下当前训练的模型



