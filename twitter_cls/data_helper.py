"""
@file   : data_helper.py
@time   : 2025-12-21
"""
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, tokenizer, is_train=True):
        self.reviews = data['text'].tolist()
        self.is_train = is_train
        if self.is_train:
            self.labels = data['target'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        text = self.reviews[item]
        input_data = self.tokenizer(text)
        # print(input_data)
        # {'input_ids': [101, 2523, 7676, 2523, 5401, 1456, 117, 678, 3613, 6820, 833, 1045, 7560, 102],
        #  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
        if self.is_train:
            label = int(self.labels[item])
            input_data['label'] = label
            return input_data
        else:
            return input_data




class Collater:
    def __init__(self, is_train):
        self.is_train = is_train
            
    def padding_to_max_len(self, x, max_len, padding_value=0):
        if len(x) > max_len:
            x = x[:max_len]
        else:
            x = x + [padding_value] * (max_len - len(x))
        return x

    def collate_fn(self, batch):
        # print(batch)  # [{'input_ids': xxx, 'attention_mask': xxx, 'token_type_ids': xx, 'label': 0}, {}]
        # 1. padding  2. 转成tensor
        max_len = max([len(item['input_ids']) for item in batch])
        if max_len > 512:
            max_len = 512

        all_input_ids = []
        all_token_type_ids = []
        all_attention_mask = []
        all_label = []
        for item in batch:
            input_ids = self.padding_to_max_len(item['input_ids'], max_len)
            token_type_ids = self.padding_to_max_len(item['token_type_ids'], max_len)
            attention_mask = self.padding_to_max_len(item['attention_mask'], max_len)
            all_input_ids.append(input_ids)
            all_token_type_ids.append(token_type_ids)
            all_attention_mask.append(attention_mask)
            if self.is_train:
                label = item['label']
                all_label.append(label)

        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        token_type_ids_tensor = torch.tensor(all_token_type_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(all_attention_mask, dtype=torch.long)
        if self.is_train:
            label_tensor = torch.tensor(all_label, dtype=torch.long)
            return input_ids_tensor, attention_mask_tensor, token_type_ids_tensor, label_tensor
        else:
            return input_ids_tensor, attention_mask_tensor, token_type_ids_tensor
