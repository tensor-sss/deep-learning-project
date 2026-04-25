"""
@file   : run_train.py
@time   : 2026-04-19
"""
# dataset  -> dataloader -> 模型 -> 优化器 和 损失 -> 训练 -> 验证 -> 保存模型 -> 推理测试
import pandas as pd
import torch
from config import set_args
from transformers.models.bert import BertTokenizer
from data_helper import MyDataset, collate_fn
from torch.utils.data import DataLoader
from model import Cls_Model
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from pathlib import Path


def compute_spearman(y_true, y_pred):
    # Spearman = Pearson(rank(y_true), rank(y_pred))
    true_rank = pd.Series(y_true).rank(method='average').to_numpy()
    pred_rank = pd.Series(y_pred).rank(method='average').to_numpy()

    true_rank = torch.tensor(true_rank, dtype=torch.float32)
    pred_rank = torch.tensor(pred_rank, dtype=torch.float32)
    vx = true_rank - true_rank.mean()
    vy = pred_rank - pred_rank.mean()
    denom = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum())
    if denom.item() == 0:
        return 0.0
    return ((vx * vy).sum() / denom).item()


def evaluate(model, valid_dataloader, device):
    model.eval()
    all_scores, all_labels = [], []
    for batch in valid_dataloader:
        batch = [t.to(device) for t in batch]
        with torch.no_grad():
            sent1_input_ids, sent1_attention_mask, sent2_input_ids, sent2_attention_mask, label = batch
            s1_vec = model.get_emb(sent1_input_ids, sent1_attention_mask)
            s2_vec = model.get_emb(sent2_input_ids, sent2_attention_mask)
            cos_sim = F.cosine_similarity(s1_vec, s2_vec)

            all_scores.extend(cos_sim.detach().cpu().tolist())
            all_labels.extend(label.detach().cpu().tolist())

    spearman = compute_spearman(all_labels, all_scores)
    return spearman

if __name__ == '__main__':
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(args.train_data_path)
    valid_df = pd.read_csv(args.valid_data_path)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)

    train_dataset = MyDataset(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    valid_dataset = MyDataset(valid_df, tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Cls_Model(args).to(device)

    # loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    best_spearman = float('-inf')
    save_path = Path(__file__).resolve().parent / "best_model.pt"

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch = [t.to(device) for t in batch]
            sent1_input_ids, sent1_attention_mask, sent2_input_ids, sent2_attention_mask, label = batch
            logits = model(sent1_input_ids, sent1_attention_mask, sent2_input_ids, sent2_attention_mask)
            loss = loss_func(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("epoch:{}, step:{}, loss:{:.6f}".format(epoch, step, loss.item()))

        avg_loss = epoch_loss / len(train_dataloader)
        spearman = evaluate(model, valid_dataloader, device)
        print("epoch:{}, train_loss:{:.6f}, valid_spearman:{:.6f}".format(epoch, avg_loss, spearman))

        if spearman > best_spearman:
            best_spearman = spearman
            torch.save(model.state_dict(), save_path)
            print("save best model to {}, best_spearman:{:.6f}".format(save_path, best_spearman))

    print("training done, best valid spearman:{:.6f}".format(best_spearman))
