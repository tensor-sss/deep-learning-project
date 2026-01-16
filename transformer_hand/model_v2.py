"""
@file   : model_v2.py
@time   : 2026-01-16
"""

import torch
import numpy as np
from torch import nn


def get_attention_padding_matrix(q, k):
    # q: (2, 17)  k: (2, 17)  => (2, 17, 17)
    # q: (2, 24)  k: (2, 17)  => (2, 24, 17)
    batch_size, q_len = q.size()
    batch_size, k_len = k.size()
    res = torch.eq(k, 0).unsqueeze(1)
    res = res.expand((batch_size, q_len, k_len))
    return res


def get_causal_mask(seq):
    batch_size, seq_len = seq.size()

    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    """
    tensor([[0., 1., 1.,  ..., 1., 1., 1.],
            [0., 0., 1.,  ..., 1., 1., 1.],
            [0., 0., 0.,  ..., 1., 1., 1.],
            ...,
            [0., 0., 0.,  ..., 0., 1., 1.],
            [0., 0., 0.,  ..., 0., 0., 1.],
            [0., 0., 0.,  ..., 0., 0., 0.]])
    """
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        self.vocab_embedding = nn.Embedding(vocab_size, 768, padding_idx=0)
        self.position_embedding = nn.Embedding(512, 768)

    def forward(self, input_ids):
        # print(input_ids.size())   # batch_size, max_len
        v_emb = self.vocab_embedding(input_ids)
        # print(v_emb.size())   # batch_size, max_len, hidden_size

        seq_len = input_ids.size(1)
        position = torch.arange(seq_len, dtype=torch.long)
        position = position.unsqueeze(0).expand_as(input_ids)
        p_emb = self.position_embedding(position)
        emb = v_emb + p_emb
        return emb


class MultiHead_Attention(nn.Module):
    def __init__(self):
        super(MultiHead_Attention, self).__init__()
        self.head_num = 12
        self.head_dim = 768 // self.head_num
        self.QW = nn.Linear(768, 768)
        self.KW = nn.Linear(768, 768)
        self.VW = nn.Linear(768, 768)
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, Q, K, V, mask):
        # print(Q.size())  # torch.Size([2, 259, 768])
        # print(K.size())  # torch.Size([2, 259, 768])
        # print(V.size())  # torch.Size([2, 259, 768])
        # torch.Size([2, 259, 768])
        # torch.Size([2, 15, 768])
        # torch.Size([2, 15, 768])

        residual = Q
        # head_dim = hidden_size / head_num
        batch_size, q_max_len, _ = Q.size()  # batch_size, max_len, hidden_size
        _, k_max_len, _ = K.size()

        # x: batch_size, max_len, 768
        Q = self.QW(Q)  # batch_size, max_len, 768
        K = self.KW(K)  # batch_size, max_len, 768
        V = self.VW(V)  # batch_size, max_len, 768

        # batch_size, max_len, 768 => batch_size, max_len, 12, 64
        Q = Q.view(batch_size, q_max_len, self.head_num, self.head_dim)
        # print(Q.size())  # torch.Size([2, 69, 12, 64])
        K = K.view(batch_size, k_max_len, self.head_num, self.head_dim)
        V = V.view(batch_size, k_max_len, self.head_num, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        # print(Q.size())   # torch.Size([2, 12, 69, 64])
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        dk = K.size(-1)
        attn = torch.matmul(Q, K.transpose(3, 2)) / np.sqrt(dk)
        # print(attn.size())  # torch.Size([2, 12, 69, 69])

        # print(attn.size())  # torch.Size([2, 69, 69])
        # print(mask.size())  # torch.Size([2, 69, 69])

        mask = mask.unsqueeze(1).repeat(1, self.head_num, 1, 1)
        # print(mask.size())  # torch.Size([2, 12, 69, 69])
        attn = attn.masked_fill_(mask, -1e9)  # torch.Size([2, 12, 69, 69])
        attn = torch.softmax(attn, dim=-1)  # torch.Size([2, 12, 69, 69])
        out = torch.matmul(attn, V)  # torch.Size([2, 12, 69, 69]) * torch.Size([2, 12, 69, 64])
        # print(out.size())  #  torch.Size([2, 12, 69, 64])

        out = out.permute(0, 2, 1, 3)  # torch.Size([2, 69, 12, 64])
        out = out.contiguous()
        out = out.view(batch_size, q_max_len, -1)  # torch.Size([2, 69, 768])
        x = residual + out
        x = self.layer_norm(x)
        return x


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(768, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, 768)
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = residual + x
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHead_Attention()
        self.feedforward = FeedForward()

    def forward(self, x, mask):
        x = self.multihead_attention(x, x, x, mask)
        x = self.feedforward(x)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, layers=12):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size)
        self.layer_encoder = nn.ModuleList([EncoderLayer() for _ in range(layers)])  # 12个独立的encoder

    def forward(self, encoder_input_ids, encoder_attention_mask):
        # embeddding + multi-head-attention + feedforward + multi-head-attention + feedforward + multi-head-attention + feedforward
        x = self.embedding(encoder_input_ids)
        mask_matrix = get_attention_padding_matrix(encoder_input_ids, encoder_input_ids)  # 为了算注意力的时候 不关注padding

        for layers in self.layer_encoder:
            x = layers(x, mask_matrix)
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.mask_multihead_attention = MultiHead_Attention()
        self.multihead_attention = MultiHead_Attention()

        self.feedforward = FeedForward()

    def forward(self, x, encoder_output, decoder_mask, encoder_decoder_mask):
        x = self.mask_multihead_attention(x, x, x, decoder_mask)  # 算的第一个Masked Multi-Head Attention
        x = self.multihead_attention(x, encoder_output, encoder_output, encoder_decoder_mask)
        x = self.feedforward(x)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, layers=12):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size)
        self.layer_decoder = nn.ModuleList([DecoderLayer() for _ in range(layers)])  # 12个独立的encoder

    def forward(self, encoder_input_ids, encoder_output, decoder_input_ids, decoder_attention_mask):
        x = self.embedding(decoder_input_ids)

        # 三个mask
        encoder_padding_mask_matrix = get_attention_padding_matrix(decoder_input_ids, encoder_input_ids)  # 为了算注意力的时候 不关注padding
        # print(encoder_padding_mask_matrix.size())   # torch.Size([2, 259, 15])

        decoder_padding_mask_matrix = get_attention_padding_matrix(decoder_input_ids, decoder_input_ids)
        decoder_subseq_mask_matrix = get_causal_mask(decoder_input_ids)
        decoder_mask = torch.ge(decoder_padding_mask_matrix+decoder_subseq_mask_matrix, 1)

        for layer in self.layer_decoder:
            x = layer(x, encoder_output, decoder_mask, encoder_padding_mask_matrix)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)
        self.predict_layer = nn.Linear(768, vocab_size)

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask):
        encoder_output = self.encoder(encoder_input_ids, encoder_attention_mask)

        # print(encoder_output.size())   # torch.Size([2, 17, 768]) batch_size, max_len, hidden_size   torch.Size([2, 12, 768])
        decoder_output = self.decoder(encoder_input_ids, encoder_output, decoder_input_ids, decoder_attention_mask)
        # print(decoder_output.size())  # torch.Size([2, 140, 768])

        # batch_size, max_len, hidden_size
        logits = self.predict_layer(decoder_output)   # batch_size, max_len, vocab_size
        # [START] 在 天 安 门 [END]
        return logits

