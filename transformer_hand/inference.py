"""
@file   : inference.py
@time   : 2026-01-16
"""
import json
import os
import torch
from torch import nn
from tqdm import tqdm
from config import set_args
from model import Transformer


def process_data(text, vocab2id):
    encoder_input_ids = []
    for v in text:
        idx = vocab2id.get(v, vocab2id['UNK'])
        encoder_input_ids.append(idx)

    encoder_attention_mask = [1] * len(encoder_input_ids)

    encoder_input_ids = torch.tensor([encoder_input_ids], dtype=torch.long)
    encoder_attention_mask = torch.tensor([encoder_attention_mask], dtype=torch.long)
    return encoder_input_ids, encoder_attention_mask


if __name__ == '__main__':
    args = set_args()
    vocab2id = json.load(open(args.vocab2id_path, 'r', encoding='utf8'))

    id2vocab = {idx: vocab for vocab, idx in vocab2id.items()}
    model = Transformer(len(vocab2id))
    model.load_state_dict(torch.load('./output/epoch0_model.bin', map_location='cpu'))  # 先把模型加载到cpu

    if torch.cuda.is_available():
        model.cuda()


    input_text = "你好吗"

    # 第一步: 数据处理:
    encoder_input_ids, encoder_attention_mask = process_data(input_text, vocab2id)

    encoder_output = model.encoder(encoder_input_ids, encoder_attention_mask)


    decoder_input_ids = [vocab2id['START']]
    decoder_attention_mask = [1] * len(decoder_input_ids)

    decoder_input_ids = torch.tensor([decoder_input_ids], dtype=torch.long)
    decoder_attention_mask = torch.tensor([decoder_attention_mask], dtype=torch.long)

    max_len = 20
    for i in range(max_len):
        out = model.decoder(encoder_input_ids, encoder_output, decoder_input_ids, decoder_attention_mask)
        logits = model.predict_layer(out)
        # print(logits.size())  # torch.Size([1, 1, 3824])

        logits = logits[:, -1]

        # 找出概率最大的那个词
        preds = torch.argmax(logits, dim=-1)  # [B, T-1]
        if preds[0] == vocab2id['END']:
            break

        preds = preds.unsqueeze(0)  # (1, 1)
        # [[2]] + [[4]] => [[2, 4]]
        decoder_input_ids = torch.cat([decoder_input_ids, preds], dim=1)
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        # print(decoder_input_ids)
        # print(decoder_attention_mask)

    s = ''
    for x in decoder_input_ids.cpu().numpy()[0]:
        s += id2vocab[x]
    print(s)
