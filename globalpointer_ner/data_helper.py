"""
@file   : data_helper.py
@time   : 2026-03-01
"""
import numpy as np
import torch
import json
from torch.utils.data import Dataset





class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id):
        self.data = data   # [{'token_list': xxx, "label_list": xx}, {xxxx}, {xxxx}]
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        cur_data = self.data[item]
        # print(cur_data)
        # {'text': ['1', '9', '9', '2', '年', '毕', '业', '于', '中', '国', '人', '民', '大', '学', '会', '计', '系', '，', '获', '学', '士', '学', '位', '；'], 'label': [['ORG', 8, 16], ['EDU', 19, 22]]}

        token_list = cur_data["text"]
        input_ids = [self.tokenizer.cls_token_id]
        for token in token_list:
            idx = self.tokenizer.convert_tokens_to_ids(token)
            input_ids.append(idx)
        input_ids = input_ids[:511] + [self.tokenizer.sep_token_id]

        label_list = cur_data['label']
        new_label_list = []
        for label in label_list:
            ent_type = label[0]
            start = label[1] + 1
            end = label[2] + 1
            ent_type_idx = self.label2id.get(ent_type)
            if end < 510 and start < 510:
                temp = [ent_type_idx, start, end]
                new_label_list.append(temp)
        return {"input_ids": input_ids, "entities": new_label_list, "token_list": ["CLS"] + token_list + ['SEP']}


def padding_to_max_len(idx, max_len, padding = 0):
    if len(idx) > max_len:
        idx = idx[:max_len]
    else:
        idx = idx + (max_len - len(idx)) * [padding]
    return idx


def collate_fn(batch):
    # [{"input_ids": input_ids, "entities": new_label_list}, {"input_ids": input_ids, "entities": new_label_list}]
    max_len = max([len(item['input_ids']) for item in batch])
    if max_len > 512:
        max_len = 512

    input_ids_list = []    # input_ids, attention_mask, token_type_ids
    attention_mask_list = []
    label_list = []
    all_token_list = []
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = [1] * len(input_ids)

        input_ids = padding_to_max_len(input_ids, max_len, padding = 0)
        attention_mask = padding_to_max_len(attention_mask, max_len, padding = 0)

        # label这块的处理 []
        labels = np.zeros((8, max_len, max_len))

        for ent in item['entities']:  # [[1, 10,13], [1,11,15], [4, 34, 37]]
            ent_type = ent[0]
            start = ent[1]
            end = ent[2]
            labels[ent_type, start, end] = 1

        label_list.append(labels.tolist())
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

        all_token_list.append(item.get('token_list'))  # 不需要padding 也不需要转tensor


    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
    label_ids = torch.tensor(label_list, dtype=torch.long)
    return input_ids, attention_mask, label_ids, all_token_list











