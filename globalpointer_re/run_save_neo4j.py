"""
@file   : run_save_neo4j.py
@time   : 2026-03-28
"""
import json
import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher


def load_data():
    data = json.load(open('./data/triples_train.json', 'r', encoding='utf8'))
    all_data = []
    for item in data:
        for spo in item['spo_list']:
            sub = spo[0]
            rel = spo[1]
            obj = spo[2]
            all_data.append([sub, rel, obj])

    df = pd.DataFrame(all_data, columns=['start', 'relation', 'end'])
    return df


class DataToNeo4j:
    def __init__(self):
        link = Graph()
        self.graph = link

        self.start = 'start'
        self.end = 'end'

        self.graph.delete_all()   # 将之前的图  全部删除
        self.matcher = NodeMatcher(link)   # 为了查找

    def create_node(self, start, end):
        # 创建节点
        for name in start:
            node = Node(self.start, name=name)
            self.graph.create(node)

        for name in end:
            node = Node(self.end, name=name)
            self.graph.create(node)

    def create_relation(self, df_data):
        m = 0
        for m in range(0, len(df_data)):
            try:
                rel = Relationship(
                    self.matcher.match(self.start).where('_.name=' + "'" + df_data['start'][m] + "'").first(),
                    df_data['relation'][m],
                    self.matcher.match(self.end).where('_.name=' + "'" + df_data['end'][m] + "'").first()
                )
                self.graph.create(rel)
            except AttributeError as e:
                print(e, m)


def data_extraction(df_data):
    node_start = []
    for i in df_data['start'].tolist():
        node_start.append(i)

    node_end = []
    for i in df_data['end'].tolist():
        node_end.append(i)

    # 去重
    node_start = list(set(node_start))
    node_end = list(set(node_end))
    return node_start, node_end


if __name__ == '__main__':
    # df_data = load_data()
    # df_data.to_csv('./data.csv', index=False)

    df_data = pd.read_csv("./data.csv")

    # print(df_data.head())
    node_start, node_end = data_extraction(df_data)

    # 创建图
    create_data = DataToNeo4j()
    # 节点
    create_data.create_node(node_start, node_end)
    # 关系
    create_data.create_relation(df_data)