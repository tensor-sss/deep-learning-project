"""
@file   : run_data_process.py
@time   : 2026-04-19
"""
import pandas as pd


def load_data(path):
    all_data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            if len(line) == 3:
                text1 = line[0]
                text2 = line[1]
                label = int(line[2])
                all_data.append([text1, text2, label])
    df = pd.DataFrame(all_data, columns=['sent1', 'sent2', 'label'])
    return df


if __name__ == '__main__':
    train_df = load_data('./data/ATEC.train.data')
    valid_df = load_data('./data/ATEC.valid.data')
    print("训练集:", train_df.shape)
    print("验证集:", valid_df.shape)

    train_df.to_csv("./data/train_data.csv", index=False)
    valid_df.to_csv("./data/valid_data.csv", index=False)







