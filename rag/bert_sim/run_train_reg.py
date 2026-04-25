"""
@file   : run_train.py
@time   : 2026-04-19
"""
# dataset  -> dataloader -> 模型 -> 优化器 和 损失 -> 训练 -> 验证 -> 保存模型 -> 推理测试
import pandas as pd
import numpy as np
import torch.cuda
from config import set_args
from transformers.models.bert import BertTokenizer, BertModel
from data_helper import MyDataset, collate_fn
from torch.utils.data import DataLoader
from model import Reg_Model
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW


# 斯皮尔曼相关系数
def evaluate(model, valid_dataloader):
    all_preds = []
    all_labels = []
    for batch in valid_dataloader:
        if torch.cuda.is_available():
            batch = [t.cuda() for t in batch]
        with torch.no_grad():
            sent1_input_ids, sent1_attention_mask, sent2_input_ids, sent2_attention_mask, label = batch
            sent1_vec = model(sent1_input_ids, sent1_attention_mask)
            sent2_vec = model(sent2_input_ids, sent2_attention_mask)
            sim_score = F.cosine_similarity(sent1_vec, sent2_vec)
            all_preds.extend(sim_score.detach().cpu().tolist())
            all_labels.extend(label.detach().cpu().tolist())

    pred_rank = pd.Series(all_preds).rank(method='average').to_numpy(dtype=np.float64)
    label_rank = pd.Series(all_labels).rank(method='average').to_numpy(dtype=np.float64)
    spearman = np.corrcoef(pred_rank, label_rank)[0, 1]
    return float(spearman)


if __name__ == '__main__':
    args = set_args()
    train_df = pd.read_csv(args.train_data_path)
    valid_df = pd.read_csv(args.valid_data_path)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)

    train_dataset = MyDataset(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    valid_dataset = MyDataset(valid_df, tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Reg_Model(args)
    if torch.cuda.is_available():
        model = model.cuda()

    # 分类
    loss_func = nn.MSELoss()   #  回归方式
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch = [t.cuda() for t in batch]
            sent1_input_ids, sent1_attention_mask, sent2_input_ids, sent2_attention_mask, label = batch
            sent1_vec = model(sent1_input_ids, sent1_attention_mask)
            sent2_vec = model(sent2_input_ids, sent2_attention_mask)
            out = F.cosine_similarity(sent1_vec, sent2_vec)
            loss = loss_func(out, label.to(dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch:{}, step:{}, loss:{}".format(epoch, step, loss.item()))

        model.eval()
        spearman = evaluate(model, valid_dataloader)
        print("epoch:{}, step:{}, valid_spearman:{:.6f}".format(epoch, step, spearman))

    save_model_path = getattr(args, 'save_model_path', './reg_last_model.pt')
    torch.save(model.state_dict(), save_model_path)
    print("last model saved to {}".format(save_model_path))
