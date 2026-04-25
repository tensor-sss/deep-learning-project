"""
@file   : run_train_tower.py
@time   : 2026-04-25
"""
# dataset -> dataloader -> 双塔模型 -> 对比学习损失 -> 训练 -> 验证 -> 保存最优模型
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pathlib import Path
from transformers.models.bert import BertTokenizer

from config_tower import set_args
from data_helper_tower import TowerDataset, collate_fn_tower
from model_tower import TowerModel


def compute_contrastive_loss(q_emb, p_emb, n_emb, negative_valid_mask, temperature=0.05):
    # q_emb: [B, H], p_emb: [B, H], n_emb: [B, N, H]
    pos_logits = torch.sum(q_emb * p_emb, dim=1, keepdim=True) / temperature
    neg_logits = torch.einsum('bh,bnh->bn', q_emb, n_emb) / temperature
    # padded negatives are masked out and do not contribute to loss
    neg_logits = neg_logits.masked_fill(negative_valid_mask < 0.5, -1e9)
    logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, 1 + N]
    labels = torch.zeros(q_emb.size(0), dtype=torch.long, device=q_emb.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
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
            (
                query_input_ids,
                query_attention_mask,
                positive_input_ids,
                positive_attention_mask,
                negative_input_ids,
                negative_attention_mask,
                negative_valid_mask,
            ) = batch

            q_emb = model.encode(query_input_ids, query_attention_mask)
            p_emb = model.encode(positive_input_ids, positive_attention_mask)
            batch_size, num_negatives, neg_len = negative_input_ids.size()
            n_input_ids = negative_input_ids.view(batch_size * num_negatives, neg_len)
            n_attention_mask = negative_attention_mask.view(batch_size * num_negatives, neg_len)
            n_emb = model.encode(n_input_ids, n_attention_mask).view(batch_size, num_negatives, -1)
            loss, logits = compute_contrastive_loss(q_emb, p_emb, n_emb, negative_valid_mask, temperature)

            pred_idx = torch.argmax(logits, dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
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
            (
                query_input_ids,
                query_attention_mask,
                positive_input_ids,
                positive_attention_mask,
                negative_input_ids,
                negative_attention_mask,
                negative_valid_mask,
            ) = batch

            q_emb = model.encode(query_input_ids, query_attention_mask)
            p_emb = model.encode(positive_input_ids, positive_attention_mask)
            batch_size, num_negatives, neg_len = negative_input_ids.size()
            n_input_ids = negative_input_ids.view(batch_size * num_negatives, neg_len)
            n_attention_mask = negative_attention_mask.view(batch_size * num_negatives, neg_len)
            n_emb = model.encode(n_input_ids, n_attention_mask).view(batch_size, num_negatives, -1)
            loss, _ = compute_contrastive_loss(q_emb, p_emb, n_emb, negative_valid_mask, args.temperature)

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
