"""
@file   : run_train.py
@time   : 2026-01-11
"""
# 加载数据 -> Dataset -> DataLoader -> 模型 -> 损失函数 -> 优化器 -> 写训练过程 -> 写验证过程(保存模型) -> 推理脚本
import json
import os
import torch
from torch import nn
from tqdm import tqdm
from config import set_args
from data_helper import load_data, MyDataset, collate_fn
from torch.utils.data import DataLoader
# from model import Transformer
from model_v2 import Transformer
from torch.optim import AdamW


def calc_loss(output, decoder_input_ids):
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    # 损失 [START] 在 北 京 [END]
    #       在     北 京 [END]
    logits = output[:, :-1, :].contiguous()
    labels = decoder_input_ids[:, 1:].contiguous()
    # print(logits.size())  # batch_size, max_len, vocab_size
    # print(labels.size())  # batch_size, max_len
    batch_size, max_len, vocab_size = logits.size()

    logits = logits.view(-1, vocab_size)  # torch.Size([1750, 4791])
    target = labels.reshape(-1)  # torch.Size([2, 874])
    # print(logits)
    # print(target)
    # print(logits.size())   # batch_size*max_len, vocab_size
    # print(target.size())   # batch_size*max_len
    loss = loss_func(logits, target)
    return loss



def evaluate(model, test_dataloader, device):
    model.eval()
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating...'):
            encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask = batch

            # print(encoder_input_ids.size(), encoder_attention_mask.size())  # torch.Size([2, 17]) torch.Size([2, 17])
            # print(decoder_input_ids.size(), decoder_attention_mask.size())  # torch.Size([2, 158]) torch.Size([2, 158])
            logits = model(encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask)
            # print(out.size())  # torch.Size([2, 125, 3824])

            # GPT 自回归：预测下一个 token
            logits = logits[:, :-1, :]  # [B, T-1, V]
            labels = decoder_input_ids[:, 1:]  # [B, T-1]
            loss_mask = decoder_attention_mask[:, 1:]  # [B, T-1]

            # 预测 token
            preds = torch.argmax(logits, dim=-1)  # [B, T-1]

            # 只在有效位置统计
            correct = (preds == labels) * loss_mask
            total_correct += correct.sum().item()
            total_count += loss_mask.sum().item()

    acc = total_correct / (total_count + 1e-8)
    return acc


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # 1. 加载数据
    train_data = load_data(args.train_data_path)
    test_data = load_data(args.test_data_path)
    print('数据样例:', train_data[0])
    print("训练集数量:", len(train_data))
    print("测试集数量:", len(test_data))
    # [{'question': xxx, "answer": xx}, {}, {}]
    vocab2id = json.load(open(args.vocab2id_path, 'r', encoding='utf8'))
    print("词表大小:", len(vocab2id))

    # 2. 实现Dataset  DataLoader
    train_dataset = MyDataset(train_data, vocab2id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = MyDataset(test_data, vocab2id)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Transformer(len(vocab2id))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = (t.to(device) for t in batch)
            encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask = batch

            # print(encoder_input_ids.size(), encoder_attention_mask.size())  # torch.Size([2, 17]) torch.Size([2, 17])
            # print(decoder_input_ids.size(), decoder_attention_mask.size())  # torch.Size([2, 158]) torch.Size([2, 158])
            out = model(encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask)
            # print(out.size())  # torch.Size([2, 125, 3824])

            loss = calc_loss(out, decoder_input_ids)
            # print(loss)  # tensor(8.3690, grad_fn=<NllLossBackward0>)
            print("epoch:{}, step:{}, loss:{:.8f}".format(epoch, step, loss))
            optimizer.zero_grad()  # 先清空优化器
            loss.backward()  # 反向求梯度
            optimizer.step()  # 把梯度更新到参数上去

        test_acc = evaluate(model, test_dataloader, device)
        # os.path.join(args.output_dir, 'log.txt')   # ./output/log.txt

        save_log_path = args.output_dir + '/' + 'log.txt'
        f = open(save_log_path, 'a', encoding='utf8')
        s = "epoch:{}, test_acc:{:.8f}\n".format(epoch, test_acc)
        f.write(s)
        f.close()

        save_model_path = args.output_dir + '/' + "epoch{}_model.bin".format(epoch)
        torch.save(model.state_dict(), save_model_path)



