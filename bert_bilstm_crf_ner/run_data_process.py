"""
@file   : run_data_process.py
@time   : 2026-03-01
"""
import json


def load_data(path):
    all_data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        token_list = []
        label_list = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                item = {"token_list": token_list, "label_list": label_list}
                all_data.append(item)
                token_list = []
                label_list = []
            else:
                vocab, label = line.split(' ')
                token_list.append(vocab)
                label_list.append(label)
    return all_data


def write_data(data, save_path):
    with open(save_path, 'w', encoding='utf8') as f:
        for item in data:
            s = json.dumps(item, ensure_ascii=False)
            f.write(s + '\n')

if __name__ == '__main__':
    # 1. 加载数据
    train_data = load_data('./data/train.txt')
    write_data(train_data, './data/train_data.json')

    dev_data = load_data("./data/dev.txt")
    write_data(dev_data, './data/dev_data.json')

    # 2. 构造标签映射
    all_label = []
    for item in train_data:
        for label in item['label_list']:
            if label not in all_label:
                all_label.append(label)
    label2id = {"O": 0}
    all_label.remove("O")

    for i, label in enumerate(all_label):
        label2id[label] = i + 1
    json.dump(label2id, open("./data/label2id.json", 'w', encoding='utf8'), ensure_ascii=False)





