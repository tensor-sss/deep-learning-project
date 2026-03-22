"""
@file   : get_rel2id.py
@time   : 2026-03-21
"""
import json

data = json.load(open('53_schemas.json', 'r', encoding='utf8'))

res = set()
for item in data:
    predicate = item['predicate']
    res.add(predicate)

tag_list = sorted(list(res))
rel2id = {tag:i for i, tag in enumerate(tag_list)}
json.dump(rel2id, open("./rel2id.json", 'w', encoding='utf8'), ensure_ascii=False)



