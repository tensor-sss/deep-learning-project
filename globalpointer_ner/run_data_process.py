"""
@file   : run_data_process.py
@time   : 2026-03-01
"""
# 英语实体   interesting => inte rest ing
import json
from seqeval.metrics.sequence_labeling import get_entities

# {'text': xxxxxxxxx, "entities": {"人名": ["张三", "李四"], "学历": ['高中', '硕士'], ...}}
def load_data(path):
    all_data = []
    all_label = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            line = json.loads(line)
            entities = get_entities(line['label_list'])

            result = []
            for ent in entities:
                ent_type = ent[0]
                start = ent[1]
                end = ent[2]
                temp = [ent_type, start, end]
                result.append(temp)
                all_label.append(ent_type)
            all_data.append({"text": line['token_list'], "label": result})

    all_label = sorted(list(set(all_label)))
    return all_data, all_label


if __name__ == '__main__':
    train_data, train_label = load_data('./data/train_data.json')
    dev_data, _ = load_data('./data/dev_data.json')

    label2id = {}
    for i, label in enumerate(train_label):
        label2id[label] = i

    json.dump(train_data, open('./data/train_data_new.json', 'w', encoding='utf8'), ensure_ascii=False)
    json.dump(dev_data, open('./data/dev_data_new.json', 'w', encoding='utf8'), ensure_ascii=False)
    json.dump(label2id, open('./data/label2id.json', 'w', encoding='utf8'), ensure_ascii=False)






