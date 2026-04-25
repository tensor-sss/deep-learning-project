"""
@file   : data_helper_tower.py
@time   : 2026-04-25
"""
import json
import torch
from torch.utils.data import Dataset


class TowerDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=64, num_negatives=10):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_negatives = num_negatives

        with open(data_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                query = str(obj['query']).strip()
                positive = str(obj.get('positive', obj.get('answer', ''))).strip()
                negatives = obj.get('negative', [])

                if not isinstance(negatives, list):
                    negatives = []

                negatives = [str(x).strip() for x in negatives if str(x).strip()]
                valid_count = min(len(negatives), self.num_negatives)
                if len(negatives) < self.num_negatives:
                    negatives = negatives + [''] * (self.num_negatives - len(negatives))
                else:
                    negatives = negatives[:self.num_negatives]
                negative_valid_mask = [1] * valid_count + [0] * (self.num_negatives - valid_count)

                if query and positive:
                    self.samples.append((query, positive, negatives, negative_valid_mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, positive, negatives, negative_valid_mask = self.samples[idx]

        q_encode = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
        )
        p_encode = self.tokenizer(
            positive,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
        )
        neg_encodes = []
        for neg in negatives:
            neg_encode = self.tokenizer(
                neg,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_attention_mask=True,
            )
            neg_encodes.append(neg_encode)

        return {
            'query_input_ids': q_encode['input_ids'],
            'query_attention_mask': q_encode['attention_mask'],
            'positive_input_ids': p_encode['input_ids'],
            'positive_attention_mask': p_encode['attention_mask'],
            'negative_input_ids_list': [x['input_ids'] for x in neg_encodes],
            'negative_attention_mask_list': [x['attention_mask'] for x in neg_encodes],
            'negative_valid_mask': negative_valid_mask,
        }


def padding_to_max_len(x, max_len, padding_value=0):
    if len(x) > max_len:
        x = x[:max_len]
    else:
        x = x + [padding_value] * (max_len - len(x))
    return x


def collate_fn_tower(batch):
    q_max_len = max([len(x['query_input_ids']) for x in batch])
    p_max_len = max([len(x['positive_input_ids']) for x in batch])
    n_max_len = max(
        [
            len(neg_input_ids)
            for x in batch
            for neg_input_ids in x['negative_input_ids_list']
        ]
    )
    num_negatives = len(batch[0]['negative_input_ids_list'])

    all_q_input_ids = []
    all_q_attention_mask = []
    all_p_input_ids = []
    all_p_attention_mask = []
    all_n_input_ids = []
    all_n_attention_mask = []
    all_n_valid_mask = []

    for item in batch:
        q_input_ids = item['query_input_ids']
        q_attention_mask = item['query_attention_mask']
        q_input_ids = padding_to_max_len(q_input_ids, q_max_len)
        q_attention_mask = padding_to_max_len(q_attention_mask, q_max_len)

        p_input_ids = item['positive_input_ids']
        p_attention_mask = item['positive_attention_mask']
        p_input_ids = padding_to_max_len(p_input_ids, p_max_len)
        p_attention_mask = padding_to_max_len(p_attention_mask, p_max_len)

        n_input_ids_list = []
        n_attention_mask_list = []
        for i in range(num_negatives):
            neg_input_ids = item['negative_input_ids_list'][i]
            neg_attention_mask = item['negative_attention_mask_list'][i]
            neg_input_ids = padding_to_max_len(neg_input_ids, n_max_len)
            neg_attention_mask = padding_to_max_len(neg_attention_mask, n_max_len)
            n_input_ids_list.append(neg_input_ids)
            n_attention_mask_list.append(neg_attention_mask)

        all_q_input_ids.append(q_input_ids)
        all_q_attention_mask.append(q_attention_mask)
        all_p_input_ids.append(p_input_ids)
        all_p_attention_mask.append(p_attention_mask)
        all_n_input_ids.append(n_input_ids_list)
        all_n_attention_mask.append(n_attention_mask_list)
        all_n_valid_mask.append(item['negative_valid_mask'])

    query_input_ids = torch.tensor(all_q_input_ids, dtype=torch.long)
    query_attention_mask = torch.tensor(all_q_attention_mask, dtype=torch.long)
    positive_input_ids = torch.tensor(all_p_input_ids, dtype=torch.long)
    positive_attention_mask = torch.tensor(all_p_attention_mask, dtype=torch.long)
    negative_input_ids = torch.tensor(all_n_input_ids, dtype=torch.long)
    negative_attention_mask = torch.tensor(all_n_attention_mask, dtype=torch.long)
    negative_valid_mask = torch.tensor(all_n_valid_mask, dtype=torch.float32)

    return (
        query_input_ids,
        query_attention_mask,
        positive_input_ids,
        positive_attention_mask,
        negative_input_ids,
        negative_attention_mask,
        negative_valid_mask,
    )
