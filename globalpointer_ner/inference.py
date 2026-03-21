"""
@file   : inference.py
@time   : 2026-03-01
"""
import json
import torch
import numpy as np
from config import set_args
from model import Model
from transformers.models.bert import BertTokenizer

# args = set_args()
# tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain)
# 原始bert分词器的特点: (1)对于中文 单个字成为一个token。(2) 对于英文 子词成为一个token
# text = "享受国务院特殊津贴Arctic Winter Games close in Whitehorse after week blending sport, culture and Indigenous games"
# res = tokenizer.tokenize(text)
# print(res)

def predict(model, tokenizer, text):
    token_list = list(text)
    input_ids = [tokenizer.cls_token_id]
    for token in token_list:
        idx = tokenizer.convert_tokens_to_ids(token)
        input_ids.append(idx)
    input_ids = input_ids[:511] + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)

    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long)

    logits = model(input_ids_tensor, attention_mask_tensor)
    # print(logits.size())   # batch_size, ent_num, max_len, max_len

    logit = logits[0]   # ent_num, max_len, max_len

    result = {}
    for i in range(logit.size(0)):
        ent_type = id2label[i]
        result[ent_type] = []
        logit_ = logit[i]
        # print(logit_.size())   # max_len, max_len

        # (max_len, max_len)   找出哪个位置的值p   大于0.5
        start_list, end_list = np.where(logit_.detach().cpu().numpy() >= 0.5)
        # print(start_list)
        # print(end_list)
        for start, end in zip(start_list, end_list):
            ent = token_list[start:end+1]
            ent = ''.join(ent)
            result[ent_type].append(ent)
    return result  # {"人名": ["张三", '王五'], '学历': ['本科'], "": [x, ,x, ,x ]}


if __name__ == '__main__':
    args = set_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label2id = json.load(open(args.label2id_path, 'r', encoding='utf8'))
    id2label = {int(idx): label for label, idx in label2id.items()}

    label_num = len(label2id)
    # 模型 + 加载训练好的权重
    model = Model(label_num)
    model.load_state_dict(torch.load('./output/epoch1_model.bin', map_location='cpu'))
    model.to(device)
    text = "吴重阳，中国国籍，大学本科，教授级高工，享受国务院特殊津贴，历任邮电部侯马电缆厂仪表试制组长、光缆分厂副厂长、研究所副所长，获得过山西省科技先进工作者、邮电部成绩优异高级工程师等多种荣誉称号。"

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain)
    res = predict(model, tokenizer, text)
    print(res)



