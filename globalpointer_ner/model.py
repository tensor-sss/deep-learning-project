"""
@file   : model.py
@time   : 2026-03-01
"""
import torch
import math
from torch import nn
from config import set_args
from transformers.models.bert import BertModel


args = set_args()



def get_table(n_position, hidden_size):
    position = torch.arange(0, n_position).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000) / hidden_size))
    embedding_tables = torch.zeros(n_position, hidden_size)
    embedding_tables[:, 0::2] = torch.sin(position * div_term)
    embedding_tables[:, 1::2] = torch.cos(position * div_term)
    return embedding_tables


class ROPEPosition(nn.Module):
    def __init__(self, max_position, embedding_size):
        super().__init__()
        position_embedding = get_table(max_position, embedding_size)
        self.cos_position = position_embedding[:, 1::2].repeat_interleave(2, dim=-1)
        self.sin_position = position_embedding[:, 0::2].repeat_interleave(2, dim=-1)


    def forward(self, qw):
        seq_len = qw.size(-2)
        qw2 = torch.stack([-qw[..., 1::2], qw[...,0::2]], dim=-1).reshape_as(qw)
        # print(self.cos_position.size())  # torch.Size([512, 384])  torch.Size([128, 768])
        # print(qw.size())  # torch.Size([128, 768])
        out = qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]
        return out


def get_attention_padding_matrix(q, k):
    batch_size, q_len = q.size()
    _, k_len = k.size()
    res = torch.eq(k, 0).unsqueeze(1)
    res = res.expand((batch_size, q_len, k_len))
    return res

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        pass


    def forward(self, y_pred, y_true):
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


class GlobalPointer(nn.Module):
    def __init__(self, head_num):
        super(GlobalPointer, self).__init__()
        # Q K
        self.head_num = head_num
        self.head_dim = 64
        self.linear = nn.Linear(768, self.head_num * self.head_dim * 2)

        self.rope = ROPEPosition(max_position=512, embedding_size=self.head_dim)


    def forward(self, inputs, mask=None):
        # inputs: batch_size, max_len, hidden_size.     batch_size, max_len, head_num, head_dim
        sequence_output = self.linear(inputs)  # batch_size, max_len, x
        # print(sequence_output.size())
        # batch_size, max_len, head_num, head_dim*2
        batch_size, max_len, _ = sequence_output.size()

        sequence_output = sequence_output.view(batch_size, max_len, self.head_num, -1)   # batch_size, max_len, self.head_num, 128

        qw = sequence_output[..., :self.head_dim]
        kw = sequence_output[..., self.head_dim:]
        # print(qw.size())   # batch_size, max_len, self.head_num, 64
        # print(kw.size())  # batch_size, max_len, self.head_num, 64
        qw = self.rope(qw)
        kw = self.rope(kw)

        qw = qw.permute(0, 2, 1, 3)  # batch_size, self.head_num, max_len, 64
        kw = kw.permute(0, 2, 1, 3)   # batch_size, self.head_num, max_len, 64
        kw = kw.transpose(-1, -2)   # batch_size, self.head_num, 64, max_len

        logits = torch.matmul(qw, kw)
        if mask is not None:
            # mask: (batch_size, max_len, max_len) =>
            # (batch_size, 1, max_len, max_len) =>
            # (batch_size, self.head_num, max_len, max_len)
            mask = mask.unsqueeze(1).repeat(1, self.head_num, 1, 1)
            logits.masked_fill_(mask.bool(), -1e9)

        # 下三角mask
        tril_mask = torch.tril(torch.ones_like(logits), -1)
        # print(tril_mask.size())  # torch.Size([32, 8, 137, 137])
        logits.masked_fill_(tril_mask.bool(), -1e9)
        return logits


class Model(nn.Module):
    def __init__(self, entity_num):
        super(Model, self).__init__()
        self.entity_num = entity_num
        self.bert = BertModel.from_pretrained(args.bert_pretrain, torch_dtype=torch.float32)  # 有了bert模型 而且加载了预训练权重
        self.globalpointer = GlobalPointer(self.entity_num)
        self.loss = MyLoss()


    def forward(self, input_ids, attention_mask, label_ids=None):
        attention_mask_matrix_padding = get_attention_padding_matrix(input_ids, input_ids)
        # attention_mask: batch_size, max_len
        # attention_mask_matrix_padding: batch_size, max_len, max_len
        out = self.bert(input_ids, attention_mask)
        last_hidden_state = out.last_hidden_state
        logits = self.globalpointer(last_hidden_state, attention_mask_matrix_padding)

        if label_ids is not None:
            loss = self.loss(logits, label_ids)
            return loss, logits
        return logits
