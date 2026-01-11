"""
@file   : data_helper.py
@time   : 2026-01-11
"""
import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


def load_data(path):
    all_data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            line = json.loads(line)
            all_data.append(line)
    return all_data


class MyDataset(Dataset):
    def __init__(self, data, vocab2id):
        self.data = data
        self.vocab2id = vocab2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        cur_data = self.data[item]
        # print(cur_data)
        # {'question': '你为什么这么感谢xxxx总？',
        #  'answer': '哎呀我的天哪，感谢我xxxx总的提督，感谢我xxxx总。小布现在在大连，再次我xxxx总，再次感谢xxxx总。'}
        question = cur_data['question']
        answer = cur_data['answer']
        encoder_input_ids = []
        for v in question:   # 你为什么这么感谢xxxx总？
            idx = self.vocab2id.get(v, self.vocab2id['UNK'])
            encoder_input_ids.append(idx)

        decoder_input_ids = []
        for v in answer:   #  哎呀我的天哪，感谢我xxxx总的提督，感谢我xxxx总。小布现在在大连，再次我xxxx总，再次感谢xxxx总。
            idx = self.vocab2id.get(v, self.vocab2id['UNK'])
            decoder_input_ids.append(idx)

        decoder_input_ids = [self.vocab2id['START']] + decoder_input_ids + [self.vocab2id['END']]
        return {"encoder_input_ids": encoder_input_ids, "decoder_input_ids": decoder_input_ids}


def padding_to_max_len(x, max_len, padding_value=0):
    if len(x) > max_len:
        x = x[:max_len]
    else:
        x = x + [padding_value] * (max_len - len(x))
    return x


def collate_fn(batch):
    #  [{"encoder_input_ids": encoder_input_ids, "decoder_input_ids": decoder_input_ids},
    #   {"encoder_input_ids": encoder_input_ids, "decoder_input_ids": decoder_input_ids}]
    encoder_max_len = max([len(item['encoder_input_ids']) for item in batch])
    decoder_max_len = max([len(item['decoder_input_ids']) for item in batch])

    if encoder_max_len >  512:
        encoder_max_len = 512

    if decoder_max_len > 512:
        decoder_max_len = 512

    all_encoder_input_ids = []
    all_encoder_attention_mask = []
    all_decoder_input_ids = []
    all_decoder_attention_mask = []
    for item in batch:
        encoder_input_ids = item['encoder_input_ids']
        encoder_attention_mask = [1] * len(encoder_input_ids)

        decoder_input_ids = item['decoder_input_ids']
        decoder_attention_mask = [1] * len(decoder_input_ids)

        encoder_input_ids = padding_to_max_len(encoder_input_ids, max_len=encoder_max_len)
        encoder_attention_mask = padding_to_max_len(encoder_attention_mask, max_len=encoder_max_len)

        decoder_input_ids = padding_to_max_len(decoder_input_ids, max_len=decoder_max_len)
        decoder_attention_mask = padding_to_max_len(decoder_attention_mask, max_len=decoder_max_len)

        all_encoder_input_ids.append(encoder_input_ids)
        all_encoder_attention_mask.append(encoder_attention_mask)

        all_decoder_input_ids.append(decoder_input_ids)
        all_decoder_attention_mask.append(decoder_attention_mask)

    encoder_input_ids = torch.tensor(all_encoder_input_ids, dtype=torch.long)
    encoder_attention_mask = torch.tensor(all_encoder_attention_mask, dtype=torch.long)
    decoder_input_ids = torch.tensor(all_decoder_input_ids, dtype=torch.long)
    decoder_attention_mask = torch.tensor(all_decoder_attention_mask, dtype=torch.long)
    return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask

