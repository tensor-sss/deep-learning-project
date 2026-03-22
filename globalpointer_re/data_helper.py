"""
@file   : data_helper.py
@time   : 2026-03-21
"""
import torch
from torch.utils.data import Dataset


class REDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]



class Collator:
    def __init__(self, tokenizer, rel2id):
        self.tokenizer = tokenizer
        self.rel2id = rel2id

    def search(self, tokens, sub_tokens):
        for i in range(len(tokens)):
            if tokens[i:i+len(sub_tokens)] == sub_tokens:
                return i
        return -1

    def padding_labels_to_max(self, x, padding_value=(0, 0)):
        max_len = max([len(i) for i in x])
        result = []
        for item in x:
            item = list(item)
            new_item = item + [padding_value] * (max_len - len(item))
            result.append(new_item)
        return result

    def padding_batch_to_max(self, x):
        # x: list => [tensor, tensor, tensor]  # (44, ?, 2)
        max_len = max([item.size(1) for item in x])

        result = []
        for item in x:
            rel_num, _, se_p = item.size()
            cur_len = item.size(1)
            if cur_len < max_len:
                pad_max = max_len - cur_len
                pad_matrix = torch.zeros(size=(rel_num, pad_max, se_p))
                new_item = torch.cat([item, pad_matrix], dim=1)
            else:
                new_item = item

            result.append(new_item)
        # (4, 44, 5, 2)
        result = torch.stack(result)
        return result

    def padding_to_max(self, x, padding_value=0):
        max_len = max([len(i) for i in x])
        result = []
        for item in x:
            item = item + [padding_value] * (max_len - len(item))
            result.append(item)
        res = torch.tensor(result, dtype=torch.long)
        return res

    def __call__(self, batch):
        # self.tokenizer
        all_input_ids_list = []
        all_attention_mask_list = []
        all_head_labels_list = []
        all_tail_labels_list = []
        all_entity_labels_list = []
        all_tokens_list = []
        for i, item in enumerate(batch):
            text = item['text']

            # 对text分词
            tokens = self.tokenizer.tokenize(text)
            tokens = tokens[:512-2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            # [x, ,x ,x ,x ]
            # [a ,a , ]
            # [b, b,
            spo_list = item['spo_list']  # [[s, r, o], [], []]
            spoes = []
            for s, p, o in spo_list:
                # "失眠症", "辅助治疗", "引导意象和冥想"
                # print(s, p, o)
                p = self.rel2id.get(p)  # 辅助治疗 => 35

                s_token = self.tokenizer.tokenize(s)
                o_token = self.tokenizer.tokenize(o)

                s_idx = self.search(tokens, s_token)
                o_idx = self.search(tokens, o_token)
                if s_idx != -1 and o_idx != -1:
                    # 头实体的 开始和结束    关系   尾实体的 开始和结束
                    spoes.append([s_idx, s_idx+len(s_token)-1, p, o_idx, o_idx+len(o_token)-1])


            # 三组数据
            head_labels = [set() for _ in range(len(self.rel2id))]
            tail_labels = [set() for _ in range(len(self.rel2id))]
            entity_labels = [set() for _ in range(2)]   # [{(sh, st), (sh, st)}, {(oh, ot)}]

            for sh, st, p, oh, ot in spoes:
                head_labels[p].add((sh, oh))
                tail_labels[p].add((st, ot))
                entity_labels[0].add((sh, st))
                entity_labels[1].add((oh, ot))

            # head_labels [[], [], [(sh, oh), (sh, oh)], [], [(sh, st)], [], []]
            # tail_labels [[], [], [(oh, ot), (oh, ot], [], [(oh, ot)], [], []]
            # entity_labels [[(sh, st), (sh, st)], [(oh, ot), (oh, ot)...]]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            all_input_ids_list.append(input_ids)
            all_attention_mask_list.append(attention_mask)


            head_labels = self.padding_labels_to_max(head_labels, (0, 0))
            tail_labels = self.padding_labels_to_max(tail_labels, (0, 0))
            entity_labels = self.padding_labels_to_max(entity_labels, (0, 0))

            head_labels = torch.tensor(head_labels, dtype=torch.long)   # (44, ?, 2)
            tail_labels = torch.tensor(tail_labels, dtype=torch.long)   # (44, ?, 2)
            entity_labels = torch.tensor(entity_labels, dtype=torch.long)   # (2, ?, 2)

            all_head_labels_list.append(head_labels)
            all_tail_labels_list.append(tail_labels)
            all_entity_labels_list.append(entity_labels)

            all_tokens_list.append(tokens)


        head_labels_tensor = self.padding_batch_to_max(all_head_labels_list)
        tail_labels_tensor = self.padding_batch_to_max(all_tail_labels_list)
        entity_labels_tensor = self.padding_batch_to_max(all_entity_labels_list)

        input_ids_tensor = self.padding_to_max(all_input_ids_list)
        attention_mask_tensor = self.padding_to_max(all_attention_mask_list)
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "head_labels": head_labels_tensor,
            "tail_labels": tail_labels_tensor,
            "entity_labels": entity_labels_tensor,
            "tokens_list": all_tokens_list
        }









