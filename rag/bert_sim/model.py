"""
@file   : model.py
@time   : 2026-04-19
"""
import torch
from torch import nn
from transformers.models.bert import BertModel


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrain_model)


    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)

        # 第一种: 取cls做线性变换后的向量
        out_vec = output.pooler_output
        # print(out_vec.size())   # torch.Size([4, 768])
        return out_vec

        # 第二种: 取所有向量 做平均池化       注意: 要排除padding的影响
        # last_hidden_state = output.last_hidden_state  # batch_size, max_len, hidden_size
        # torch.mean(last_hidden_state, dim=1)   # batch_size, hidden_size








