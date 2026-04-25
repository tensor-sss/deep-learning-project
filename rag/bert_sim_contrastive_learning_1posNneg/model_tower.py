"""
@file   : model_tower.py
@time   : 2026-04-25
"""
import torch
from torch import nn
from transformers.models.bert import BertModel


class TowerModel(nn.Module):
    def __init__(self, args):
        super(TowerModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrain_model).to(dtype=torch.float32)

    def encode(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emb = output.pooler_output
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def forward(self, query_input_ids, query_attention_mask, answer_input_ids, answer_attention_mask):
        q_emb = self.encode(query_input_ids, query_attention_mask)
        a_emb = self.encode(answer_input_ids, answer_attention_mask)
        return q_emb, a_emb
