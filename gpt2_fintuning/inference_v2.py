"""
@file   : inference_v2.py
@time   : 2026-01-11
"""
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./gpt2_pretrain')
model = GPT2LMHeadModel.from_pretrained('./gpt2_pretrain')

question = '我爱你'
# 第一步: [CLS] 我 爱 你 [SEP]
# 解码
input_ids_title = tokenizer.encode(question, add_special_tokens=False)  # [x, x, x]
input_ids = [tokenizer.cls_token_id] + input_ids_title + [tokenizer.sep_token_id]
input_ids = torch.tensor([input_ids], dtype=torch.long)  # []

max_len = 10
for i in range(max_len):
    # [101, xxx, x, 102]
    out = model(input_ids)
    logits = out.logits
    # print(logits.size())   # batch_size, max_len, vocab_size

    last_logits = logits[:, -1, :]   #  1, vocab_size
    preds = torch.argmax(last_logits, dim=-1, keepdim=True)  # [B, T-1]

    if preds[0][0] == tokenizer.sep_token_id:
        break
    # print(input_ids)  # tensor([[ 101, 2769, 4263,  872,  102]])
    # print(preds)   # tensor([[100]])

    input_ids = torch.cat([input_ids, preds], dim=1)
    # print(input_ids)   # tensor([[ 101, 2769, 4263,  872,  102, 100]])

input_ids = input_ids[0]
res = tokenizer.decode(input_ids)
res = res.split('[SEP]')[-1]
print(res)
