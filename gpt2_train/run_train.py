"""
@file   : run_train.py
@time   : 2026-01-03
"""
# 加载数据 -> 实现dataset -> Dataloader -> 模型 ->  优化器 -> 损失函数 -> 训练过程 -> 验证过程 -> inference
import torch
from torch import nn
from config import set_args
from torch.utils.data import DataLoader
from torch.optim import AdamW
from data_helper import load_data, MyDataset, collate_fn
from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2Config
import swanlab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

swanlab.login(api_key="6YRUZelhDiHwXBHvOUbwl", save=True)

def get_model(args):
    # 有预训练权重
    model = GPT2LMHeadModel.from_pretrained(args.gpt2_pretrain)  # 没有.bin文件 所以会报错 找不到权重

    # 没有预训练权重  使用别人配置文件 做一个随机初始化的GPT2模型
    # config = GPT2Config.from_pretrained('./gpt2_pretrain/config.json')
    # model = GPT2LMHeadModel(config=config)
    return model


def calc_loss(logits, label, loss_mask):
    # logits: batch_size, max_len, vocab_size
    # label: batch_size, max_len

    logits = logits[:, :-1, :].contiguous()
    label = label[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:]

    loss_func = nn.CrossEntropyLoss(reduction='none', ignore_index=0)    # sum  mean
    # print(logits.size())   # batch_size, max_len, vocab_size
    # print(loss_mask.size())   # batch_size, max_len

    batch_size, max_len, vocab_size = logits.size()
    logits = logits.view(-1, vocab_size)
    label = label.reshape(-1)

    loss_mask = loss_mask.reshape(-1)
    # print(logits.size())    # (1036, vocab_size)
    # print(loss_mask.size())   # (1036,)

    loss = loss_func(logits, label)
    loss = (loss * loss_mask).mean()
    return loss


def evaluate(model, test_dataloader):
    model.eval()

    total_correct = 0
    total_count = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, loss_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            loss_mask = loss_mask.to(device)

            outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            logits = outputs.logits  # [B, T, V]

            # GPT 自回归：预测下一个 token
            logits = logits[:, :-1, :]  # [B, T-1, V]
            labels = input_ids[:, 1:]  # [B, T-1]
            loss_mask = loss_mask[:, 1:]  # [B, T-1]

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

    swanlab.init(
        project='GPT2-generation',
        experiment_name='作文生成'
    )

    # 1. 加载数据
    train_data = load_data(args.train_data_path)
    test_data = load_data(args.test_data_path)
    print("训练集:", len(train_data))
    print("测试集:", len(test_data))

    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_pretrain)   # 实例化一个分词器

    # 2. 实现dataset + dataloader
    train_dataset = MyDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = MyDataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. 模型
    model = get_model(args)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            input_ids, attention_mask, loss_mask = batch
            # print(input_ids.size())   # torch.Size([2, 571])
            # print(attention_mask.size())  # torch.Size([2, 571])
            # print(loss_mask.size())   # torch.Size([2, 571])
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            loss_mask = loss_mask.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            # print(logits.size())   # torch.Size([2, 518, 21129])  # batch_size, max_len, vocab_size

            loss = calc_loss(logits, input_ids, loss_mask)
            # print(loss)
            # 考虑  模型最后怎么评估效果？？？
            # 1. 预测的准确率
            # 2. ？？？ ppl  rouge  bleu
            print("epoch:{}, step:{}, loss:{:.8f}".format(epoch, step, loss))
            optimizer.zero_grad()  # 先清空优化器
            loss.backward()  # 反向求梯度
            optimizer.step()  # 把梯度更新到参数上去
            swanlab.log({"train_loss": loss.item()}, step=global_step)

        test_acc = evaluate(model, test_dataloader)
        # os.path.join(args.output_dir, 'log.txt')   # ./output/log.txt
        swanlab.log({"accuracy": test_acc}, step=epoch)

        save_log_path = args.output_dir + '/' + 'log.txt'
        f = open(save_log_path, 'a', encoding='utf8')
        s = "epoch:{}, test_acc:{:.8f}\n".format(epoch, test_acc)
        f.write(s)
        f.close()

        save_model_path = args.output_dir + '/' + "epoch{}_model.bin".format(epoch)
        torch.save(model.state_dict(), save_model_path)



