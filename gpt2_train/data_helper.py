"""
@file   : data_helper.py
@time   : 2026-01-03
"""
import json
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

        # res = self.tokenizer
        # print(res)

        # x, x, x, x,   y, y, y, y, y, y, y

