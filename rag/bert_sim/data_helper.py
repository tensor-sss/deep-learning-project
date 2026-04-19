"""
@file   : data_helper.py
@time   : 2026-04-19
"""
import torch
from torch.utils.data import Dataset



class MyDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.sent1_list = dataframe['sent1'].tolist()
        self.sent2_list = dataframe['sent2'].tolist()
        self.label_list = dataframe['label'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sent1_list)

    def __getitem__(self, idx):
        sent1 = self.sent1_list[idx]
        sent2 = self.sent2_list[idx]
        label = self.label_list[idx]

        sent1_input_ids = self.tokenizer.encode(sent1)
        sent2_input_ids = self.tokenizer.encode(sent2)

        return {"sent1_input_ids": sent1_input_ids, 'sent2_input_ids': sent2_input_ids, 'label': int(label)}


def padding_to_max_len(x, max_len, padding_value=0):
    if len(x) > max_len:
        x = x[:max_len]
    else:
        x = x + [padding_value] * (max_len - len(x))
    return x


def collate_fn(batch):
    sent1_max_len = max([len(x['sent1_input_ids']) for x in batch])
    sent2_max_len = max([len(x['sent2_input_ids']) for x in batch])

    sent1_max_len = min(sent1_max_len, 512)
    sent2_max_len = min(sent2_max_len, 512)

    all_sent1_input_ids = []
    all_sent1_attention_mask = []

    all_sent2_input_ids = []
    all_sent2_attention_mask = []

    all_labels = []
    for item in batch:
        sent1_input_ids = item['sent1_input_ids']
        sent1_attention_mask = [1] * len(sent1_input_ids)
        sent1_input_ids = padding_to_max_len(sent1_input_ids, max_len=sent1_max_len)
        sent1_attention_mask = padding_to_max_len(sent1_attention_mask, max_len=sent1_max_len)

        sent2_input_ids = item['sent2_input_ids']
        sent2_attention_mask = [1] * len(sent2_input_ids)
        sent2_input_ids = padding_to_max_len(sent2_input_ids, max_len=sent2_max_len)
        sent2_attention_mask = padding_to_max_len(sent2_attention_mask, max_len=sent2_max_len)

        all_sent1_input_ids.append(sent1_input_ids)
        all_sent1_attention_mask.append(sent1_attention_mask)

        all_sent2_input_ids.append(sent2_input_ids)
        all_sent2_attention_mask.append(sent2_attention_mask)

        all_labels.append(item['label'])


    sent1_input_ids = torch.tensor(all_sent1_input_ids, dtype=torch.long)
    sent1_attention_mask = torch.tensor(all_sent1_attention_mask, dtype=torch.long)
    sent2_input_ids = torch.tensor(all_sent2_input_ids, dtype=torch.long)
    sent2_attention_mask = torch.tensor(all_sent2_attention_mask, dtype=torch.long)

    label = torch.tensor(all_labels, dtype=torch.float)

    return sent1_input_ids, sent1_attention_mask, sent2_input_ids, sent2_attention_mask, label














