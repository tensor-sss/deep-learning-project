import json
import torch
from torch.utils.data import Dataset


def load_data(path):
    all_data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = json.loads(line)   # json.dumps
            question = line['question']
            answer = line['answer']
            all_data.append({"question": question, "answer": answer})
    return all_data


class MyDatast(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['question']
        target_text = item['answer']
        encoder_input_ids = self.tokenizer.encode(input_text)
        encoder_input_ids = encoder_input_ids[:-1]

        decoder_input_ids = self.tokenizer.encode(target_text)
        #  START x  x x x x END
        #   x    x  x x x END
        decoder_input_ids = [self.tokenizer.convert_tokens_to_ids('<start>')] + decoder_input_ids  # end
        labels = decoder_input_ids[1:]  #  x x x x END
        decoder_input_ids = decoder_input_ids[:-1]  # START x x x x
        return {
            'encoder_input_ids': encoder_input_ids,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels
        }


def padding_to_max(x, max_len, pad_value=0):
    if len(x) >= max_len:
        return x[:max_len]
    else:
        return x + [pad_value] * (max_len - len(x))


def collate_fn(batch):
    encoder_max_len = max([len(item['encoder_input_ids']) for item in batch])
    decoder_max_len = max([len(item['decoder_input_ids']) for item in batch])

    if encoder_max_len > 512:
        encoder_max_len = 512

    if decoder_max_len > 512:
        decoder_max_len = 512

    encoder_input_ids_list = []
    encoder_attention_mask_list = []

    decoder_input_ids_list = []
    decoder_attention_mask_list = []

    labels_list = []
    for item in batch:
        encoder_input_ids = item['encoder_input_ids']
        decoder_input_ids = item['decoder_input_ids']
        labels = item['labels']

        encoder_attention_mask = [1] * len(encoder_input_ids)
        decoder_attention_mask = [1] * len(decoder_input_ids)

        encoder_input_ids = padding_to_max(encoder_input_ids, encoder_max_len, pad_value=0)
        encoder_attention_mask = padding_to_max(encoder_attention_mask, encoder_max_len, pad_value=0)

        decoder_input_ids = padding_to_max(decoder_input_ids, decoder_max_len, pad_value=0)
        decoder_attention_mask = padding_to_max(decoder_attention_mask, decoder_max_len, pad_value=0)

        labels = padding_to_max(labels, decoder_max_len, pad_value=-100)

        encoder_input_ids_list.append(encoder_input_ids)
        encoder_attention_mask_list.append(encoder_attention_mask)
        decoder_input_ids_list.append(decoder_input_ids)
        decoder_attention_mask_list.append(decoder_attention_mask)
        labels_list.append(labels)

    encoder_input_ids = torch.tensor(encoder_input_ids_list, dtype=torch.long)
    encoder_attention_mask = torch.tensor(encoder_attention_mask_list, dtype=torch.long)
    decoder_input_ids = torch.tensor(decoder_input_ids_list, dtype=torch.long)
    decoder_attention_mask = torch.tensor(decoder_attention_mask_list, dtype=torch.long)
    labels = torch.tensor(labels_list, dtype=torch.long)

    return {
        'encoder_input_ids': encoder_input_ids,
        'encoder_attention_mask': encoder_attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'decoder_attention_mask': decoder_attention_mask,
        'labels': labels
    }




