"""
@file   : run_train_tower.py
@time   : 2026-04-25
"""
# dataset -> dataloader -> 双塔模型 -> 对比学习损失 -> 训练 -> 验证 -> 保存最优模型
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pathlib import Path
from transformers.models.bert import BertTokenizer

from config_tower import set_args
from data_helper_tower import TowerDataset, collate_fn_tower
from model_tower import TowerModel


def compute_contrastive_loss(q_emb, a_emb, temperature=0.05):
    logits = torch.matmul(q_emb, a_emb.t()) / temperature
    labels = torch.arange(q_emb.size(0), device=q_emb.device)

    loss_q2a = nn.CrossEntropyLoss()(logits, labels)
    loss_a2q = nn.CrossEntropyLoss()(logits.t(), labels)
    loss = (loss_q2a + loss_a2q) / 2.0
    return loss, logits


def evaluate(model, valid_dataloader, device, temperature):
    model.eval()
    total_loss = 0.0
    total_batch = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in valid_dataloader:
            batch = [t.to(device) for t in batch]
            query_input_ids, query_attention_mask, answer_input_ids, answer_attention_mask = batch

            q_emb, a_emb = model(query_input_ids, query_attention_mask, answer_input_ids, answer_attention_mask)
            loss, logits = compute_contrastive_loss(q_emb, a_emb, temperature)

            pred_idx = torch.argmax(logits, dim=1)
            labels = torch.arange(logits.size(0), device=logits.device)
            correct = (pred_idx == labels).sum().item()

            total_loss += loss.item()
            total_batch += 1
            total_correct += correct
            total_samples += logits.size(0)

    avg_loss = total_loss / max(total_batch, 1)
    recall_at_1 = total_correct / max(total_samples, 1)
    return avg_loss, recall_at_1


if __name__ == '__main__':
    args = set_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)

    train_dataset = TowerDataset(args.train_data_path, tokenizer, max_len=args.max_len)
    valid_dataset = TowerDataset(args.valid_data_path, tokenizer, max_len=args.max_len)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_tower,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_tower,
    )

    model = TowerModel(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    best_recall_at_1 = float('-inf')
    save_path = Path(args.save_model_path)
    if not save_path.is_absolute():
        save_path = Path(__file__).resolve().parent / save_path

    print('train_size:{}, valid_size:{}, device:{}'.format(len(train_dataset), len(valid_dataset), device))

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            batch = [t.to(device) for t in batch]
            query_input_ids, query_attention_mask, answer_input_ids, answer_attention_mask = batch

            q_emb, a_emb = model(query_input_ids, query_attention_mask, answer_input_ids, answer_attention_mask)
            loss, _ = compute_contrastive_loss(q_emb, a_emb, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print('epoch:{}, step:{}, loss:{:.6f}'.format(epoch, step, loss.item()))

        train_avg_loss = epoch_loss / max(len(train_dataloader), 1)
        valid_loss, valid_recall = evaluate(model, valid_dataloader, device, args.temperature)

        print(
            'epoch:{}, train_loss:{:.6f}, valid_loss:{:.6f}, valid_recall@1:{:.6f}'.format(
                epoch,
                train_avg_loss,
                valid_loss,
                valid_recall,
            )
        )

        if valid_recall > best_recall_at_1:
            best_recall_at_1 = valid_recall
            torch.save(model.state_dict(), save_path)
            print('save best model to {}, best_recall@1:{:.6f}'.format(save_path, best_recall_at_1))

    print('training done, best valid recall@1:{:.6f}'.format(best_recall_at_1))
