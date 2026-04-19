"""
@file   : run_train.py
@time   : 2026-04-19
"""
# dataset  -> dataloader -> 模型 -> 优化器 和 损失 -> 训练 -> 验证 -> 保存模型 -> 推理测试
import pandas as pd
import torch.cuda
from config import set_args
from transformers.models.bert import BertTokenizer, BertModel
from data_helper import MyDataset, collate_fn
from torch.utils.data import DataLoader
from model import Model
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW


# 斯皮尔曼相关系数  皮尔逊相关系数
def evaluate(model, valid_dataloader):
    for batch in valid_dataloader:
        if torch.cuda.is_available():
            batch = [t.cuda() for t in batch]
        with torch.no_grad():
            sent1_input_ids, sent1_attention_mask, sent2_input_ids, sent2_attention_mask, label = batch
            sent1_vec = model(sent1_input_ids, sent1_attention_mask)
            sent2_vec = model(sent2_input_ids, sent2_attention_mask)

        sent1_vec = F.normalize(sent1_vec, dim=-1, eps=1e-12)
        sent2_vec = F.normalize(sent2_vec, dim=-1, eps=1e-12)
        # (4, 768),  (4, 768)  =>  (4, 1)
        out = sent1_vec * sent2_vec
        out = out.sum(dim=-1)
        print(out)
        exit()

        # out = F.sigmoid(out)
        # 以0.5分界线 去看



if __name__ == '__main__':
    args = set_args()
    train_df = pd.read_csv(args.train_data_path)
    valid_df = pd.read_csv(args.valid_data_path)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)

    train_dataset = MyDataset(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    valid_dataset = MyDataset(valid_df, tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Model(args)

    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.BCEWithLogitsLoss()  # sigmoid + cross_entropy
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch = [t.cuda() for t in batch]
            sent1_input_ids, sent1_attention_mask, sent2_input_ids, sent2_attention_mask, label = batch
            sent1_vec = model(sent1_input_ids, sent1_attention_mask)
            sent2_vec = model(sent2_input_ids, sent2_attention_mask)
            sent1_vec = F.normalize(sent1_vec, dim=-1, eps=1e-12)
            sent2_vec = F.normalize(sent2_vec, dim=-1, eps=1e-12)
            # (4, 768),  (4, 768)  =>  (4, 1)
            out = sent1_vec * sent2_vec
            out = out.sum(dim=-1)
            loss = loss_func(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch:{}, step:{}, loss:{}".format(epoch, step, loss.item()))

            model.eval()
            evaluate(model, valid_dataloader)







