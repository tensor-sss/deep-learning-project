"""
@file   : ROPE.py
@time   : 2026-03-13
"""
import torch
import math
from torch import nn


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


if __name__ == '__main__':
    rope = ROPEPosition(512, 768)

    qw = torch.randn(size=(128, 768))
    out1 = rope(qw)
    print(out1.size())  # torch.Size([128, 768])

    kw = torch.randn(size=(128, 768))
    out2 = rope(kw)
    print(out2.size())  # torch.Size([128, 768])

    res = torch.matmul(out1, out2.transpose(-1, -2))
    print(res.size())   # torch.Size([128, 128])

