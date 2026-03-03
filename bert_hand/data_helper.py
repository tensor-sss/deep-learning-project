import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    # 传入词表方便对照词表进行转id
    def __init__(self, data, vocab2id):
        self.data = data
        self.vocab2id = vocab2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        cur_data = self.data[item]

        input_ids = []
        for v in cur_data['tokens_list']:
            idx = self.vocab2id.get(v, self.vocab2id['UNK'])
            input_ids.append(idx)

        mask_label = []
        for v in cur_data['mask_label']:
            idx = self.vocab2id.get(v, self.vocab2id['UNK'])
            mask_label.append(idx)

        return {'input_ids': input_ids,
                'segment_ids': cur_data['segment_ids'],
                'mask_position': cur_data['mask_position'],
                'mask_label': mask_label,
                'label': cur_data['label']}

def padding_to_max_len(x, max_len, padding_value = 0):
    if len(x) > max_len:
        x = x[:max_len]
    else:
        x = x + [padding_value] * (max_len - len(x))
    return x

def collate_fn(batch):
    max_len1 = max([len(item['input_ids']) for item in batch])
    max_len2 = max([len(item['mask_label'])for item in batch])

    # 对数据进行进一步处理
    input_ids_list = []
    token_type_ids_list = []
    mask_position_list = []
    mask_label_list = []
    # label_list = []
    # 需要一个注意力掩码列表来对是否需要进行注意力计算进行区分
    attention_mask_list = []

    for item in batch:
        attention_mask = [1] * len(item['input_ids'])
        attention_mask = padding_to_max_len(attention_mask, max_len1)

        input_ids = padding_to_max_len(item['input_ids'], max_len1)

        token_type_ids = padding_to_max_len(item['segment_ids'], max_len1)

        mask_position = padding_to_max_len(item['mask_position'], max_len2)
        mask_label = padding_to_max_len(item['mask_label'], max_len2)
        # 收集label
        # label_list.append(item['label'])
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
        mask_position_list.append(mask_position)
        mask_label_list.append(mask_label)

    # 对label直接采用列表推导式
    label = torch.tensor([item['label'] for item in batch])
    # 转张量
    input_ids = torch.tensor(input_ids_list)
    token_type_ids = torch.tensor(token_type_ids_list)
    attention_mask = torch.tensor(attention_mask_list)
    mask_position = torch.tensor(mask_position_list)
    mask_label = torch.tensor(mask_label_list)
    return input_ids, attention_mask, token_type_ids, mask_position, mask_label, label
















