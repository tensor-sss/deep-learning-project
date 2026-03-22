"""
@file   : inference.py
@time   : 2026-03-01
"""
import torch
import json
from config import set_args
from model import Model
from transformers.models.bert import BertTokenizer
from seqeval.metrics.sequence_labeling import get_entities


def get_entity(pred_labels, text):
    result = []
    s = ''
    i = 0
    for vocab, label in zip(text, pred_labels):
        if label == 'O':
            if i != 0 and len(pred_labels[i - 1].split('-')) >= 2:
                label = pred_labels[i - 1].split('-')[1]
            if len(s) != 0:
                result.append(s + '|' + label)
            s = ''

        elif label[0] == 'B':
            if i != 0 and len(pred_labels[i - 1].split('-')) >= 2:
                label = pred_labels[i - 1].split('-')[1]
            if len(s) != 0:
                result.append(s + "|" + label)
            s = vocab
        else:
            s += vocab
        i += 1

    final_result = {}
    for item in result:
        if len(item) != 0:
            name, tag = item.split("|")
            if tag in final_result:
                final_result[tag].append(name)
            else:
                final_result[tag] = [name]
    return final_result


def predict_text(text, tokenizer, model):
    input_ids = [tokenizer.cls_token_id]
    for token in text:
        idx = tokenizer.convert_tokens_to_ids(token)
        input_ids.append(idx)

    input_ids = input_ids[:511]
    input_ids = input_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)

    # input_ids: [x, x, x]  attention_mask: [1, 1, 1, ..]
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)

    logits = model(input_ids, attention_mask)
    # print(logits)
    # print(logits.size())   # torch.Size([1, 30, 17])

    # CLS xxxxx SEP
    # 解出每个位置真正预测的标签
    logits = logits[0][1:-1]   # 把cls sep直接过滤掉
    _, pred_label = torch.max(logits, dim=-1)  # 概率分布

    result = []
    for label in pred_label.numpy():  # pred_label.numpy() 把tensor转成numpy数组
        tag = id2label.get(label)
        result.append(tag)

    # final_res = get_entity(result, text)
    result = get_entities(result)
    # print(result)  # [('NAME', 0, 1), ('PRO', 5, 9), ('ORG', 15, 18), ('ORG', 24, 27)]
    final_res = {}
    for tag, start, end in result:
        ent = text[start:end+1]
        if tag in final_res:
            final_res[tag].append(ent)
        else:
            final_res[tag] = [ent]
    return final_res


if __name__ == '__main__':
    args = set_args()
    label2id = json.load(open(args.label2id_path, 'r', encoding='utf8'))
    # {"O": 0, "B-P", 1, ...}
    id2label = {idx: label  for label, idx in label2id.items()}
    # {0: "O", 1: "B-P", ...}
    label_num = len(label2id)
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain)

    model = Model(label_num)
    model.load_state_dict(torch.load('./output/epoch3_model.bin', map_location='cpu'))

    # text = '马先生1978年毕业于伦敦大学生物化工专业，获学士学位。'
    # 李四读的是计算机专业，20年从北京大学毕业。 在华为公司上班。
    while True:
        text = input("输入:")
        res = predict_text(text, tokenizer, model)
        print(res)   # {'NAME': ['李四'], 'PRO': ['计算机专业'], 'ORG': ['北京大学', '华为公司']}

