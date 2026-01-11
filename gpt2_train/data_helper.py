"""
@file   : data_helper.py
@time   : 2026-01-03
"""
import json
import torch
from torch.utils.data import Dataset


def load_data(path):
    all_data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # '{x: x, x:x}'
            line = json.loads(line)   # '{x: x, x:x}' => {x: x, x:x}    # json.dumps()   {x: x, x:x} => "{x: x, x:x}"
            all_data.append(line)
    return all_data


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        json_data = self.data[item]
        title = json_data['title']
        article = json_data['article']
        # print(title)    # "xxxx"
        # print(article)   # "xxxxxxxxxx"
        # title =》 idx
        # article =》 idx
        # loss_mask =>
        # {"title": "我爱你", "article": "我爱北京天安门"}
        # [BOS] 我 爱 你 我 爱 北 京 天 安 门 [EOS]
        #  0    0  0  0 1  1  1 1  1  1  1  1   # 都是要求损失

        # [BOS] 我 爱 你 我 爱 北 京 天 安 门 [EOS] [PAD] [PAD] [PAD]
        # 0    0  0  0 1  1  1 1  1  1  1   1      0     0     0
        # input_ids, loss_mask

        # CLS 我 爱 你 SEP 我 爱 北 京 天 安 门 SEP
        # CLS 我 爱 你 SEP 我 爱 北 京 天 安 门 SEP
        # 0   0  0  0  1  1  1  1  1  1 1  1  0

        input_ids_title = self.tokenizer.encode(title, add_special_tokens=False)  # [x, x, x]
        input_ids_article = self.tokenizer.encode(article, add_special_tokens=False)

        input_ids_title = [self.tokenizer.cls_token_id] + input_ids_title + [self.tokenizer.sep_token_id]
        input_ids_article = input_ids_article + [self.tokenizer.sep_token_id]
        loss_mask = [0] * (len(input_ids_title) - 1) + [1] * len(input_ids_article)  + [0]
        input_ids = input_ids_title + input_ids_article
        return {"input_ids": input_ids, "loss_mask": loss_mask}

def pad_to_max_len(x, max_len, padding_value=0):
    if len(x) > max_len:
        x = x[:max_len]
    else:
        x = x + (max_len - len(x)) * [padding_value]
    return x


def collate_fn(batch):
    # [{"input_ids": input_ids, "loss_mask": loss_mask}, {"input_ids": input_ids, "loss_mask": loss_mask}, ..]
    max_len = max([len(item['input_ids']) for item in batch])
    if max_len > 1024:
        max_len = 1024

    all_input_ids = []
    all_attention_mask = []
    all_loss_mask = []
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = pad_to_max_len([1] * len(input_ids), max_len=max_len)  # [1, 1, 1, 1, 1, 0, 0, 0]
        input_ids = pad_to_max_len(item['input_ids'], max_len=max_len)
        loss_mask = pad_to_max_len(item['loss_mask'], max_len=max_len)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_loss_mask.append(loss_mask)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    loss_mask = torch.tensor(all_loss_mask, dtype=torch.long)
    return input_ids, attention_mask, loss_mask


