"""
@file   : run_train.py
@time   : 2026-03-21
"""
import json
import torch.cuda
import numpy as np
from model import Model
from config import set_args
from torch import nn
from data_helper import REDataset, Collator
from torch.utils.data import DataLoader
from transformers.models.bert import BertTokenizer

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def calc_loss(self, y_pred, y_true):
        # y_pred.size():  batch_size, x, max_len, max_len
        # y_true.size():  batch_size, x, max_len, max_len
        batch_size = y_pred.size(0)
        entity_num = y_pred.size(1)
        y_pred = y_pred.view(batch_size * entity_num, -1)
        y_true = y_true.view(batch_size * entity_num, -1)
        # print(logits.size())   # torch.Size([256, 11025])
        # print(label_ids.size())  # torch.Size([256, 11025])

        y_pred = (1 - 2 * y_true) * y_pred  # 将y_pred中所有正样本的打分 加个负号
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12

        y_pred_neg = torch.cat([y_pred_neg, torch.ones_like(y_pred_neg[..., :1])], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, torch.ones_like(y_pred_pos[..., :1])], dim=-1)

        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss

    def _sparse_to_dense(self, outputs, labels):
        B, C, L, _ = outputs.shape
        dense_labels = torch.zeros_like(outputs)

        # 【修复点】：强制将 labels 转换为 long 类型，防止 float 索引报错
        labels = labels.long()

        # 构造用于批量索引的 batch 和 class 索引 (确保也是 long 类型)
        b_idx = torch.arange(B, dtype=torch.long, device=outputs.device).view(B, 1, 1).expand_as(labels[..., 0])
        c_idx = torch.arange(C, dtype=torch.long, device=outputs.device).view(1, C, 1).expand_as(labels[..., 0])

        i_idx = labels[..., 0]
        j_idx = labels[..., 1]

        # 创建 mask 防止 padding 越界
        mask = (i_idx >= 0) & (j_idx >= 0) & (i_idx < L) & (j_idx < L)

        # 向量化赋值
        dense_labels[b_idx[mask], c_idx[mask], i_idx[mask], j_idx[mask]] = 1.0

        return dense_labels


    def forward(self, entity_output, head_output, tail_output, entity_labels, head_labels, tail_labels):
        # print(entity_output.size())  # torch.Size([4, 2, 157, 157])
        # print(entity_labels.size())  # torch.Size([4, 2, 16, 2])
        # print(head_output.size())   # torch.Size([4, 44, 157, 157])
        # print(head_labels.size())   # torch.Size([4, 44, 12, 2])
        # 1. 稀疏标签 (Sparse) 转为 稠密标签 (Dense)
        entity_labels_dense = self._sparse_to_dense(entity_output, entity_labels)
        head_labels_dense = self._sparse_to_dense(head_output, head_labels)
        tail_labels_dense = self._sparse_to_dense(tail_output, tail_labels)

        # 2. 分别计算三部分的 GlobalPointer Loss
        entity_loss = self.calc_loss(entity_output, entity_labels_dense)
        head_loss = self.calc_loss(head_output, head_labels_dense)
        tail_loss = self.calc_loss(tail_output, tail_labels_dense)

        # 3. 聚合损失
        # calc_loss 返回的是一维 Tensor，形状为 [batch_size * classes]。通常求平均。
        total_loss = entity_loss.mean() + head_loss.mean() + tail_loss.mean()
        return total_loss


def evaluate(model, test_dataloader, device):
    for step, batch in enumerate(test_dataloader):
        batch = {k: v.to(device)  if k != 'tokens_list' else v  for k, v in batch.items()}
        # {
        #     "input_ids": input_ids_tensor,
        #     "attention_mask": attention_mask_tensor,
        #     "head_labels": head_labels_tensor,
        #     "tail_labels": tail_labels_tensor,
        #     "entity_labels": entity_labels_tensor,
        #     'tokens_list': [['C', '', ''], ['', '', ...]]
        # }
        entity_output, head_output, tail_output = model(batch['input_ids'], batch['attention_mask'])

        batch_size = entity_output.size(0)
        tokens_list = batch['tokens_list']
        for i in range(batch_size):
            cur_tokens_list = tokens_list[i]
            cur_entity_output = entity_output[i]   # (2, max_len, max_len)
            cur_head_output = head_output[i]   # (44, max_len, max_leb)
            cur_tail_output = tail_output[i]   # (44, max_len, max_leb)
            # print(cur_head_output.size())
            subject_entity_output = cur_entity_output[0]   # torch.Size([98, 98])
            object_entity_output = cur_tail_output[1]   # torch.Size([98, 98])

            sub_start_list, sub_end_list = np.where(subject_entity_output.detach().cpu().numpy() >= 0.2)
            obj_start_list, obj_end_list = np.where(object_entity_output.detach().cpu().numpy() >= 0.2)

            subject_ids = []  # [[x, x], [x, x], [x, x], [], []]
            for m, n in zip(sub_start_list, sub_end_list):
                subject_ids.append([m, n])

            object_ids = []   # [[x, x], [x, x], [], ]
            for m, n in zip(obj_start_list, obj_end_list):
                object_ids.append([m, n])

            # 先找主  然后再穷举所有的尾  最后再看 start-start end-end两个概率是否高于阈值
            spo_list = []
            for sh, st in subject_ids:
                for oh, ot in object_ids:
                    sub_ent = cur_tokens_list[sh: st+1]
                    obj_ent = cur_head_output[oh: ot+1]

                    rel1 = np.where(cur_head_output[:, sh, oh].detach().cpu().numpy() > 0.1)[0]
                    rel2 = np.where(cur_tail_output[:, st, ot].detach().cpu().numpy() > 0.1)[0]

                    res = set(rel1) & set(rel2)
                    for r in res:
                        rel = id2rel.get(r)
                        spo_list.append((sub_ent, rel, obj_ent))

# 1. 把损失看懂
# 2. 损失为啥很高
# 3. 把验证写一下
if __name__ == '__main__':
    args = set_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据 -> dataset -> dataloader
    train_data = json.load(open(args.train_data_path, 'r', encoding='utf8'))
    test_data = json.load(open(args.dev_data_path, 'r', encoding='utf8'))

    rel2id = json.load(open(args.rel2id_path, 'r', encoding='utf8'))
    id2rel = {idx: rel for rel, idx in rel2id.items()}

    train_dataset = REDataset(train_data)
    test_dataset = REDataset(test_data)

    # print(train_dataset[42])
    # {'text': 'B族链球菌感染@主要危险因素包括存在晚期肾病、神经系统疾病、恶性肿瘤和免疫抑制。', 'spo_list': [['B族链球菌感染', '高危因素', '晚期肾病'], ['B族链球菌感染', '高危因素', '神经系统疾病'], ['B族链球菌感染', '高危因素', '恶性肿瘤'], ['B族链球菌感染', '高危因素', '免疫抑制']]}
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain)

    # 钩子函数
    collate_fn = Collator(tokenizer, rel2id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = Model(len(rel2id))
    model.to(device)

    loss_func = MyLoss()   # 损失函数

    for batch in train_dataloader:
        batch = {k: v.to(device)  if k != 'tokens_list' else v  for k, v in batch.items()}
        """
        {
            "input_ids": input_ids_tensor, 
            "attention_mask": attention_mask_tensor,
            "head_labels": head_labels_tensor,
            "tail_labels": tail_labels_tensor,
            "entity_labels": entity_labels_tensor
        }
        """
        # print(batch['input_ids'].size())   # batch_size, max_len
        # print(batch['attention_mask'].size())  # batch_size, max_len
        # print(batch['head_labels'].size())   # batch_size, 44, ?, 2
        # print(batch['tail_labels'].size())   # batch_size, 44, ?, 2
        # print(batch['entity_labels'].size())  # batch_size, 2, ?, 2
        entity_output, head_output, tail_output = model(batch['input_ids'], batch['attention_mask'])

        loss = loss_func(entity_output, head_output, tail_output, batch['entity_labels'], batch['head_labels'], batch['tail_labels'])
        # print(loss)
        evaluate(model, test_dataloader, device)
        exit()







