"""
@file   : data_helper_tower.py
@time   : 2026-04-25
"""
import json
import torch
from torch.utils.data import Dataset


class TowerDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=64):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(data_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                query = str(obj['query']).strip()
                answer = str(obj['answer']).strip()
                if query and answer:
                    self.samples.append((query, answer))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, answer = self.samples[idx]

        q_encode = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
        )
        a_encode = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
        )

        return {
            'query_input_ids': q_encode['input_ids'],
            'query_attention_mask': q_encode['attention_mask'],
            'answer_input_ids': a_encode['input_ids'],
            'answer_attention_mask': a_encode['attention_mask'],
        }


def padding_to_max_len(x, max_len, padding_value=0):
    if len(x) > max_len:
        x = x[:max_len]
    else:
        x = x + [padding_value] * (max_len - len(x))
    return x


def collate_fn_tower(batch):
    q_max_len = max([len(x['query_input_ids']) for x in batch])
    a_max_len = max([len(x['answer_input_ids']) for x in batch])

    all_q_input_ids = []
    all_q_attention_mask = []
    all_a_input_ids = []
    all_a_attention_mask = []

    for item in batch:
        q_input_ids = item['query_input_ids']
        q_attention_mask = item['query_attention_mask']
        q_input_ids = padding_to_max_len(q_input_ids, q_max_len)
        q_attention_mask = padding_to_max_len(q_attention_mask, q_max_len)

        a_input_ids = item['answer_input_ids']
        a_attention_mask = item['answer_attention_mask']
        a_input_ids = padding_to_max_len(a_input_ids, a_max_len)
        a_attention_mask = padding_to_max_len(a_attention_mask, a_max_len)

        all_q_input_ids.append(q_input_ids)
        all_q_attention_mask.append(q_attention_mask)
        all_a_input_ids.append(a_input_ids)
        all_a_attention_mask.append(a_attention_mask)

    query_input_ids = torch.tensor(all_q_input_ids, dtype=torch.long)
    query_attention_mask = torch.tensor(all_q_attention_mask, dtype=torch.long)
    answer_input_ids = torch.tensor(all_a_input_ids, dtype=torch.long)
    answer_attention_mask = torch.tensor(all_a_attention_mask, dtype=torch.long)

    return query_input_ids, query_attention_mask, answer_input_ids, answer_attention_mask
