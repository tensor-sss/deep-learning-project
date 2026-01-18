
import torch
from config import set_args
from torch.utils.data import DataLoader
from data_helper import load_data, MyDatast, collate_fn
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate(model, test_dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(test_dataloader)
    return avg_loss


# 加载数据 -> 构建自己的Dataset -> DataLoader -> 写模型 -> 优化器 -> 损失函数 -> 训练过程 -> 评估过程+模型的保存 -> 推理
if __name__ == "__main__":
    args = set_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_data = load_data(args.train_data_path)
    test_data = load_data(args.test_data_path)
    print(f'训练数据条数: {len(train_data)}')
    print(f'测试数据条数: {len(test_data)}')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_path)
    model = model.to(device)

    # 构建自己Dataset -> DataLoader
    train_dataset = MyDatast(train_data, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataset = MyDatast(test_data, tokenizer)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # 不传label，就需要取logits 自己算损失
            # 如果传入了labels, loss算出损失
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

    # 评估+保存模型省略
    eval_loss = evaluate(model, test_dataloader, device)
    print(f'Epoch: {epoch}, Eval Loss: {eval_loss}')
    # 保存模型
    with open(f'{args.output_dir}/eval_loss.txt', 'a', encoding='utf8') as f:
        f.write(f'Epoch: {epoch}, Eval Loss: {eval_loss}\n')

    model.save_pretrained(f'{args.output_dir}/model_epoch_{epoch}')
    tokenizer.save_pretrained(f'{args.output_dir}/model_epoch_{epoch}')

