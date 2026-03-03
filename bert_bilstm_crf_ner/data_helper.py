"""
@file   : data_helper.py
@time   : 2026-03-01
"""
import torch
import json
from torch.utils.data import Dataset


def load_data(path):
    all_data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = json.loads(line)
            all_data.append(line)
    return all_data



class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id):
        self.data = data   # [{'token_list': xxx, "label_list": xx}, {xxxx}, {xxxx}]
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        cur_data = self.data[item]
        token_list = cur_data["token_list"]
        label_list = cur_data['label_list']

        # token_list: ['2', '0', '0', '6', '年', '3', '月', '加', '入', '原', '平', '安', '银', '行', '，', '历', '任', '运', '营', '总', '监', '、', '人', '力', '资', '源', '总', '监', '，']
        # label_list: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'B-TITLE', 'I-TITLE', 'I-TITLE', 'I-TITLE', 'O', 'B-TITLE', 'I-TITLE', 'I-TITLE', 'I-TITLE', 'I-TITLE', 'I-TITLE', 'O']

        input_ids = [self.tokenizer.cls_token_id]
        for token in token_list:
            idx = self.tokenizer.convert_tokens_to_ids(token)
            input_ids.append(idx)

        label_ids = [-100]
        for label in label_list:
            idx = self.label2id[label]
            label_ids.append(idx)

        input_ids = input_ids[:511] + [self.tokenizer.sep_token_id]
        label_ids = label_ids[:511] + [-100]  # SEP 设为 -100
        return {"input_ids": input_ids, "label_ids": label_ids}


def padding_to_max_len(idx, max_len, padding = 0):
    if len(idx) > max_len:
        idx = idx[:max_len]
    else:
        idx = idx + (max_len - len(idx)) * [padding]
    return idx


def collate_fn(batch):
    # padding + tensor
    # [{"input_ids": input_ids, "label_ids": label_ids}, {"input_ids": input_ids, "label_ids": label_ids}, ..]
    # 算当前batch中最大长度
    max_len = max([len(item['input_ids']) for item in batch])
    if max_len > 512:
        max_len = 512

    input_ids_list = []    # input_ids, attention_mask, token_type_ids
    attention_mask_list = []
    label_ids_list = []
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = [1] * len(input_ids)

        input_ids = padding_to_max_len(input_ids, max_len, padding = 0)
        attention_mask = padding_to_max_len(attention_mask, max_len, padding = 0)

        label_ids = item['label_ids']
        label_ids = padding_to_max_len(label_ids, max_len, padding = -100)

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        label_ids_list.append(label_ids)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
    label_ids = torch.tensor(label_ids_list, dtype=torch.long)
    return input_ids, attention_mask, label_ids










