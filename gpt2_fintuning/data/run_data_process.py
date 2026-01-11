"""
@file   : run_data_process.py
@time   : 2026-01-03
"""
from tqdm import tqdm


def load_data(path):
    all_data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            all_data.append(line)
    return all_data


def save_data(data, save_path):
    with open(save_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(data))

if __name__ == '__main__':
    all_data = load_data('./article.json')
    # print("总体的数据量:", len(all_data))   # 总体的数据量: 269370

    train_data = all_data[:100]
    test_data = all_data[100:150]

    save_data(train_data, save_path='./train_data.jsonl')
    save_data(test_data, save_path='./test_data.jsonl')

