import torch
from config import set_args
from torch.utils.data import DataLoader
from data_helper import load_data, MyDatast, collate_fn
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config



def evalute_model(model, test_dataloader, tokenizer, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            encoder_input_ids = batch['encoder_input_ids']
            attention_mask = batch['attention_mask']
            decoder_input_ids = batch['decoder_input_ids']
            decoder_attention_mask = batch['decoder_attention_mask']
            labels = batch['labels']

            outputs = model(
                input_ids=encoder_input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    return avg_loss


# 加载数据 -> 构建自己的Dataset -> DataLoader -> 写模型 -> 优化器 -> 损失函数 -> 训练过程 -> 评估过程+模型的保存 -> 推理
def get_model(args, device):
    # config = T5Config.from_pretrained(args.pretrained_model_path)
    # model = T5ForConditionalGeneration(config=config)   # 不加载预训练权重

    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_path)  # 加载预训练权重
    model = model.to(device)
    return model

if __name__ == '__main__':
    args = set_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_data = load_data(args.train_data_path)
    test_data = load_data(args.test_data_path)
    print(f'训练数据条数: {len(train_data)}')
    print(f'测试数据条数: {len(test_data)}')

    model = get_model(args, device)

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path)
    tokenizer.add_tokens('<start>')   # <start>
    print("此时词表大小:", len(tokenizer))   # 加了之后: 321

    model.resize_token_embeddings(len(tokenizer))   # 重置词表大小 因为加入了新的token

    # 构建Dataset -> DataLoader
    train_dataset = MyDatast(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = MyDatast(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            encoder_input_ids = batch['encoder_input_ids']
            attention_mask = batch['encoder_attention_mask']
            decoder_input_ids = batch['decoder_input_ids']
            decoder_attention_mask = batch['decoder_attention_mask']
            labels = batch['labels']

            outputs = model(
                input_ids=encoder_input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )
            loss = outputs.loss
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')
            # 反向传播和优化步骤省略
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        eval_loss = evalute_model(model, test_dataloader, tokenizer, device)
        print(f'Epoch: {epoch}, Eval Loss: {eval_loss}')
        # 保存模型
        with open(f'{args.output_dir}/eval_loss.txt', 'a', encoding='utf8') as f:
            f.write(f'Epoch: {epoch}, Eval Loss: {eval_loss}\n')

        model.save_pretrained(f'{args.output_dir}/model_epoch_{epoch}')
        tokenizer.save_pretrained(f'{args.output_dir}/model_epoch_{epoch}')












