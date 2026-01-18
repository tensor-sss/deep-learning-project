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
        question = item['question']
        answer = item['answer']
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}  # generation prompt
        ]


        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,  # 不写成False的话 会把数据直接转成id  写成False 不转id
            add_generation_prompt=True,
        )
        # 找到特殊token的位置
        # p1 = input_ids.rfind(151644)
        p1 = -1   # think
        for i in range(len(input_ids)):
            if input_ids[i] == 151667:
                p1 = i
                break
            
        p2 = -1    # im_start
        for i in range(len(input_ids)-1, -1, -1):
            if input_ids[i] == 151644:
                p2 = i
                break

        input_ids = input_ids[:p2]   # 把多余东西删掉
        labels = [-100] * p1 + input_ids[p1:]  
        '''
        <|im_start|>user
        真心话大冒险输了会怎么样？<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        正常什么什么？你跟师兄师姐打牌怎么着怎么着？真心话大冒险，大冒险就是来连线你，就是你输了就来连我呗？<|im_end|>
        <|im_start|>assistant
        '''
        return {
            "input_ids": input_ids,
            "labels": labels
        }


def padding_to_max_len(input_ids, max_len, pad_id=0):
    if len(input_ids) < max_len:
        pad_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
    else:
        input_ids = input_ids[:max_len]
    return input_ids


def collate_fn(batch):
    # batch [{"input_ids":..., "labels":...}, {...}]
    max_len = max([len(item['input_ids']) for item in batch])


    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = [1] * len(input_ids)
        labels = item['labels']

        # padding
        input_ids = padding_to_max_len(input_ids, max_len)
        attention_mask = padding_to_max_len(attention_mask, max_len, pad_id=0)
        labels = padding_to_max_len(labels, max_len, pad_id=-100)

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    batch_data = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long)
    }
    return batch_data