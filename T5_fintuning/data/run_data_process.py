"""
@file   : run_data_process.py
@time   : 2026-01-11
"""
import json
from tqdm import tqdm
from collections import Counter


def load_data(path):
    all_data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = json.loads(line)   # json.dumps
            messages = line['messages'][:2]
            if len(messages) == 2:
                question = messages[0]['content']
                answer = messages[1]['content']
                all_data.append({"question": question, "answer": answer})
    return all_data


def write_data(data, save_name):
    with open(save_name, 'w', encoding='utf8') as f:
        for item in data:
            s = json.dumps(item, ensure_ascii=False)
            f.write(s + '\n')


def build_vocab(news):
    vocabs_list = []
    for json_data in tqdm(news):
        text = json_data['question'] + json_data['answer']
        vocabs = list(text)
        vocabs_list.extend(vocabs)

    vocabs_count = dict(Counter(vocabs_list))
    vocabs_count = sorted(vocabs_count.items(), key=lambda x: x[1], reverse=True)
    vocabs_count = vocabs_count[:20000]
    vocab2id = {'PAD': 0, 'UNK': 1, 'START': 2, "END": 3}
    for i, v in enumerate(vocabs_count):
        vocab2id[v[0]] = i + 4
    return vocab2id


if __name__ == '__main__':
    path = 'question_answer.jsonl'
    all_data = load_data(path)
    # print(len(all_data))  # 81060

    train_data = all_data[:80000]
    test_data = all_data[80000:]

    write_data(train_data, 'train_data.jsonl')
    write_data(test_data, 'test_data.jsonl')





