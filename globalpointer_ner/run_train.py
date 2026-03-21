"""
@file   : run_train.py
@time   : 2026-03-01
"""

# config.py -> 加载数据 -> dataset -> dataloader -> 模型 -> 优化器 损失函数 -> 训练过程 -> 验证过程 -> 训练结束后 写推理过程inference.py
import os
import json
import torch
import numpy as np
from config import set_args
from sklearn.metrics import accuracy_score
from data_helper import NERDataset, collate_fn
from transformers.models.bert import BertTokenizer
from torch.utils.data import DataLoader
from model import Model
from torch import nn
from torch.optim import AdamW


def evaluate(dev_dataloader, model):
    a, b = 0, 0
    for step, batch in enumerate(dev_dataloader):
        # 原来的文本
        input_ids, attention_mask, label_ids = [t.to(device) for t in batch[:-1]]  # (input_ids, attention_mask, label_ids)
        all_token_list = batch[-1]

        logits = model(input_ids, attention_mask)  # model.forward(input_ids, attention_mask)
        # batch_size, 8, max_len, max_len
        batch_size = logits.size(0)
        for b in range(batch_size):
            logit = logits[b]  # ent_num, max_len, max_len  预测的
            label_id = label_ids[b]   # 真实的
            token_list = all_token_list[b]   # 辅助转回实体文本

            predict_entities = []
            true_entities = []
            for i in range(logit.size(0)):
                logit_ = logit[i]
                label_id_ = label_id[i]

                start_list, end_list = np.where(logit_.detach().cpu().numpy() >= 0.5)
                # print(start_list)
                # print(end_list)
                for start, end in zip(start_list, end_list):
                    ent = token_list[start:end + 1]
                    ent = ''.join(ent)
                    predict_entities.append(ent)

                start_list, end_list = np.where(label_id_.detach().cpu().numpy() == 1)
                for start, end in zip(start_list, end_list):
                    ent = token_list[start:end+1]
                    ent = ''.join(ent)
                    true_entities.append(ent)

            # print(predict_entities)
            # print(true_entities)  # ]
            # 预测对了多少个  在真实标签中有多少实体

            a += len(set(predict_entities) & set(true_entities))
            b += len(true_entities)
    acc = a / b
    # precision  recall  f1_score=(2*p*r)/(p+r)
    return acc


if __name__ == '__main__':
    args = set_args()

    # 创建输出文件夹
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    train_data = json.load(open(args.train_data_path, 'r', encoding='utf8'))
    dev_data = json.load(open(args.dev_data_path, 'r', encoding='utf8'))
    print("训练集样本数:", len(train_data))    # 训练集样本数: 3821
    print("验证集样本数:", len(dev_data))   # 验证集样本数: 463

    label2id = json.load(open(args.label2id_path, 'r', encoding='utf8'))
    id2label = {int(idx): label for label, idx in label2id.items()}
    label_num = len(label2id)

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain)

    # 2. 实现dataset    文本 和标签 转成对应id
    train_dataset = NERDataset(train_data, tokenizer, label2id)
    dev_dataset = NERDataset(dev_data, tokenizer, label2id)

    # 3. 实现dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)  # 封装batch 转tensor
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = Model(label_num)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    loss_func = nn.CrossEntropyLoss(ignore_index = -100)

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # input_ids, attention_mask, label_ids, token_list
            input_ids, attention_mask, label_ids = [t.to(device) for t in batch[:-1]]  # (input_ids, attention_mask, label_ids)
            # print(input_ids.size())
            # print(attention_mask.size())
            # print(label_ids.size())   # batch_size, 8, max_len, max_len
            loss, logits = model(input_ids, attention_mask, label_ids)   # model.forward(input_ids, attention_mask)
            loss = loss.mean()
            print('epoch:{}, step:{}, loss:{: 8f}'.format(epoch + 1, step + 1, loss))
            # exit()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_acc = evaluate(dev_dataloader, model)

        # 保存准确率
        save_log_path = args.output_dir + '/' + 'log.txt'
        with open(save_log_path, 'a') as f:
            f.write('epoch:{}, test_acc:{: 8f}'.format(epoch + 1, test_acc))
        # 每一轮保存一下当前训练的模型
        save_model_path = args.output_dir + '/' + 'epoch{}_model.bin'.format(epoch + 1)
        torch.save(model.state_dict(), save_model_path)


