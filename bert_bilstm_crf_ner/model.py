"""
@file   : model.py
@time   : 2026-03-01
"""
import torch
from torch import nn
from config import set_args
from torchcrf import CRF
from transformers.models.bert import BertModel


args = set_args()

# 半精度 全精度  混合精度  float32  bfloat16   float16
# BERT + 逐token分类
class Model(nn.Module):
    def __init__(self, label_num):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_pretrain, torch_dtype=torch.float32)  # 有了bert模型 而且加载了预训练权重
        self.prediction = nn.Linear(768, label_num)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)
        last_hidden_state = out.last_hidden_state
        # print(last_hidden_state.size())   # batch_size, max_len, 768
        logits = self.prediction(last_hidden_state)
        # print(logits.size())
        return logits

# BERT + BiLSTM + 逐token分类    为什么要加BiLSTM:  因为bert的位置信息比较弱
class Model_V1(nn.Module):
    def __init__(self, label_num):
        super(Model_V1, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_pretrain, torch_dtype=torch.float32)  # 有了bert模型 而且加载了预训练权重
        self.bilstm = nn.LSTM(input_size=768, hidden_size=768//2,
                              num_layers = 1, bidirectional=True, batch_first=True)
        self.prediction = nn.Linear(768, label_num)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)
        last_hidden_state = out.last_hidden_state
        # print(last_hidden_state.size())   # batch_size, max_len, 768
        last_hidden_state, _ = self.bilstm(last_hidden_state)
        # print(last_hidden_state.size())   # torch.Size([32, 57, 768])
        logits = self.prediction(last_hidden_state)
        # print(logits.size())
        return logits


# BERT + BiLSTM + CRF     CRF: 条件随机场  课后学习   常用在标签标注中。    NER     分词
# 我 喜 欢  北 京 天 安 门    # 微软的中文分词  人民日报分词语料
# B  B  I  B  I  B I  I

# HMM隐马尔科夫模型   CRF:条件随机场
class Model_V2(nn.Module):
    def __init__(self, label_num):
        super(Model_V2, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_pretrain, torch_dtype=torch.float32)  # 有了bert模型 而且加载了预训练权重
        self.bilstm = nn.LSTM(input_size=768, hidden_size=768//2,
                              num_layers = 1, bidirectional=True, batch_first=True)
        self.prediction = nn.Linear(768, label_num)
        self.crf = CRF(label_num, batch_first=True)   # batch_size, max_len, hidden_size    max_len, batch_size, hidden_size

    def forward(self, input_ids, attention_mask, label):
        out = self.bert(input_ids, attention_mask)
        last_hidden_state = out.last_hidden_state
        # print(last_hidden_state.size())   # batch_size, max_len, 768
        last_hidden_state, _ = self.bilstm(last_hidden_state)
        # print(last_hidden_state.size())   # torch.Size([32, 57, 768])
        seq_out = self.prediction(last_hidden_state)

        if label is not None:
            loss = -self.crf(seq_out, label, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            logits = self.crf.decode(seq_out, mask=attention_mask.bool())
            return logits
