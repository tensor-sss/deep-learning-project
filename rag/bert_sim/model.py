"""
@file   : model.py
@time   : 2026-04-19
"""
import torch
from torch import nn
from transformers.models.bert import BertModel


class Reg_Model(nn.Module):
    def __init__(self, args):
        super(Reg_Model, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrain_model).to(dtype=torch.float32)


    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)

        out_vec = output.pooler_output
        # print(out_vec.size())   # torch.Size([4, 768])
        return out_vec



class Cls_Model(nn.Module):
    def __init__(self, args):
        super(Cls_Model, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrain_model).to(dtype=torch.float32)
        self.linear = nn.Linear(768 * 3, 2)

    def get_emb(self, input_ids, attention):
        output = self.bert(input_ids, attention)
        emb = output.pooler_output
        return emb

    def forward(self, s1_input_ids, s1_attention_mask, s2_input_ids, s2_attention_mask):
        s1_emb = self.get_emb(s1_input_ids, s1_attention_mask)
        s2_emb = self.get_emb(s2_input_ids, s2_attention_mask)

        diff = torch.abs(s1_emb - s2_emb)
        out = torch.cat([s1_emb, s2_emb, diff], dim=1)   # batch_size, 2304
        logits = self.linear(out)
        return logits




