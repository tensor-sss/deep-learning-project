"""
@file   : model.py
@time   : 2026-03-01
"""
import torch
from torch import nn
from config import set_args
from transformers.models.bert import BertModel


args = set_args()

# 半精度 全精度  混合精度  float32  bfloat16   float16

class Model(nn.Module):
    def __init__(self, label_num):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_pretrain, dtype=torch.float32)  # 有了bert模型 而且加载了预训练权重
        self.prediction = nn.Linear(768, label_num)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)
        last_hidden_state = out.last_hidden_state
        # print(last_hidden_state.size())   # batch_size, max_len, 768
        logits = self.prediction(last_hidden_state)
        # print(logits.size())
        return logits

