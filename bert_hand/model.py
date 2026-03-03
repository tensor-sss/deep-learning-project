import torch
import numpy as np
from torch import nn

# 创建注意力掩码，padding位置为True（值为0的位置）
# k: [batch_size, seq_len]
# 返回: [batch_size, q_len, k_len]
def get_attention_padding_matrix(q, k):
    batch_size, q_len = q.size()
    _, k_len = k.size()
    res = torch.eq(k, 0).unsqueeze(1)
    res = res.expand((batch_size, q_len, k_len))
    return res

# 三种嵌入:词嵌入，位置嵌入，句段嵌入
class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        self.vocab_embedding = nn.Embedding(vocab_size, 768, padding_idx = 0)
        self.position_embedding = nn.Embedding(512, 768)
        self.segment_embedding = nn.Embedding(2, 768)

    def forward(self, input_ids, token_type_ids):
        v_emb = self.vocab_embedding(input_ids)

        seq_len = input_ids.size(1)
        position = torch.arange(seq_len, dtype = torch.long)
        position = position.unsqueeze(0).expand_as(input_ids)
        p_emb = self.position_embedding(position)

        t_emb = self.segment_embedding(token_type_ids)

        emb = v_emb + p_emb + t_emb
        return emb
# 注意力机制
class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.head_num = 12
        self.head_dim = 768 // self.head_num
        self.QW = nn.Linear(768,768)
        self.KW = nn.Linear(768,768)
        self.VW = nn.Linear(768,768)
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, x, mask):
        # 备份一份不处理的数据
        residual = x
        # s.size() batch_size, max_len, hidden_size
        batch_size, max_len, _ = x.size()

        Q = self.QW(x)
        K = self.KW(x)
        V = self.VW(x)

        Q = Q.view(batch_size, max_len, self.head_num, self.head_dim)
        K = K.view(batch_size, max_len, self.head_num, self.head_dim)
        V = V.view(batch_size, max_len, self.head_num, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        # Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
        # d_k 是 每个注意力头的维度
        # 除以 sqrt{d_k} 可以将数值拉回到均值为 0、方差为 1 的分布，使梯度更稳定
        dk = K.size(-1)
        attn = torch.matmul(Q, K.transpose(2,3)) / np.sqrt(dk)
        mask = mask.unsqueeze(1).repeat(1, self.head_num, 1, 1)
        attn = attn.masked_fill_(mask, -1e9)
        attn = torch.softmax(attn, dim = -1)
        out = torch.matmul(attn, V)

        out = out.permute(0, 2, 1, 3)
        out = out.contiguous()
        out = out.view(batch_size, max_len, -1)
        # 做残差链接
        x = residual + out
        # LayerNorm
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

class PredictLayer(nn.Module):
    def __init__(self, vocab_size):
        super(PredictLayer, self).__init__()
        # MLM预测被mask的token是什么
        self.linear1 = nn.Linear(768, vocab_size)
        # NSP预测两个句子是否是一个句子
        self.linear2 = nn.Linear(768, 2)

    def forward(self, last_all_hidden, mask_position):
        mask_position = mask_position.unsqueeze(2)
        mask_position = mask_position.expand(-1, -1, last_all_hidden.size(-1))

        mask_vector = torch.gather(last_all_hidden, 1, mask_position)

        mask_pred = self.linear1(mask_vector)
        # 只取CLS向量
        cls_vec = last_all_hidden[:, 0, :]
        nsp_pred = self.linear2(cls_vec)
        return mask_pred, nsp_pred
class BertLayer(nn.Module):
    def __init__(self):
        super(BertLayer, self).__init__()
        self.multihead_attention = SelfAttention()
        self.feedforward = FeedForward()

    def forward(self, x, mask):
        x = self.multihead_attention(x, mask)
        x = self.feedforward(x)
        return x


class BERT(nn.Module):
    def __init__(self, vocab_size, layers = 12):
        super(BERT, self).__init__()
        # 第一部分：Embedding
        self.emb = Embedding(vocab_size)
        # 第二部分：Muti-head Attention
        # 第三部分：FeedForward
        self.layer_encoder = nn.ModuleList([BertLayer() for _ in range(layers)])
        # 第四部分:predict
        self.predict = PredictLayer(vocab_size)

    def forward(self, input_ids, token_type_ids, mask_position):
        x = self.emb(input_ids, token_type_ids)
        mask_matrix = get_attention_padding_matrix(input_ids, input_ids)
        for layer in self.layer_encoder:
            x = layer(x, mask_matrix)
        mask_pred, nsp_pred = self.predict(x, mask_position)
        return mask_pred, nsp_pred















