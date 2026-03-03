import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type = str, default = './data/test.csv', help = 'test_path')
    parser.add_argument('--train_path', type= str, default = './data/train.csv', help = 'train_path')
    parser.add_argument('--test_data_path', type = str, default = './data/test_data.json', help = 'test_data_path')
    parser.add_argument('--train_data_path', type = str, default = './data/train_data.json', help = 'train_data_path')
    parser.add_argument('--vocab_path', type = str, default = './data/vocab2id.json', help = 'vocab2id')
    parser.add_argument('--batch_size', type = int, default = 2, help = 'batch_size')

    args = parser.parse_args()
    return args