"""
@file   : run_train.py
@time   : 2025-12-21
"""
import os
import torch
import pandas as pd
from torch import nn
from model import Model
from config import set_args
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from data_helper import MyDataset, Collater
from transformers import AutoTokenizer
import swanlab
# 666
swanlab.login(api_key="6YRUZelhDiHwXBHvOUbwl", save=True)


def evaluate():
    model.eval()   # 验证状态
    all_true_label = []
    all_pred_label = []
    for batch in test_dataloader:
        if torch.cuda.is_available():
            batch = [t.cuda() for t in batch]
        input_ids, attention_mask, token_type_ids, label = batch
        # print(feat.size())  # torch.Size([32, 15])
        # print(label.size())   # torch.Size([32])
        out = model(input_ids, attention_mask, token_type_ids)  # 等价 model.forward(feat)
        # print(out.size())   # torch.Size([32, 3])
        _, pred_label = torch.max(out, dim=-1)

        pred_label = pred_label.cpu().detach().numpy().tolist()
        true_label = label.cpu().detach().numpy().tolist()
        all_true_label.extend(true_label)
        all_pred_label.extend(pred_label)
    acc = accuracy_score(all_true_label, all_pred_label)
    return acc


if __name__ == '__main__':
    # 大数据   30G  100G
    # 加载数据 + dataset + dataloader + model + 损失函数 + 优化器 + 写训练的过程 + 写验证的过程 + 保存模型
    args = set_args()

    os.makedirs(args.output_dir, exist_ok=True)

    swanlab.init(
        project='twitter',
        experiment_name='bert_fintuing'
    )

    train_df = pd.read_csv(args.train_data_path)
    test_df = pd.read_csv(args.test_data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model, use_fast=True)

    collater = Collater(is_train=True)
    train_dataset = MyDataset(train_df, tokenizer, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collater.collate_fn)

    test_dataset = MyDataset(test_df, tokenizer, is_train=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collater.collate_fn)

    model = Model(num_label=2)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    loss_func = nn.CrossEntropyLoss()
    global_step = 0
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            if torch.cuda.is_available():
                batch = [t.cuda() for t in batch]
            input_ids, attention_mask, token_type_ids, label = batch
            # print(input_ids.size())   # batch_size, max_len
            # print(attention_mask.size())
            # print(token_type_ids.size())
            # print(label.size())  # batch_size
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_func(logits, label)
            print("epoch:{}, step:{}, loss:{:.8f}".format(epoch, step, loss))
            optimizer.zero_grad()  # 先清空优化器
            loss.backward()  # 反向求梯度
            optimizer.step()  # 把梯度更新到参数上去
            swanlab.log({"train_loss": loss.item()}, step=global_step)

        test_acc = evaluate()
        os.path.join(args.output_dir, 'log.txt')   # ./output/log.txt
        swanlab.log({"accuracy": test_acc}, step=epoch)

        save_log_path = args.output_dir + '/' + 'log.txt'
        f = open(save_log_path, 'a', encoding='utf8')
        s = "epoch:{}, test_acc:{:.8f}\n".format(epoch, test_acc)
        f.write(s)
        f.close()

        save_model_path = args.output_dir + '/' + "epoch{}_model.bin".format(epoch)
        torch.save(model.state_dict(), save_model_path)






















