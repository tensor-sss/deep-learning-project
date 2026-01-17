import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output', help='模型输出目录')
    parser.add_argument('--train_data_path', type=str, default='./data/train_data.jsonl', help='训练数据路径')
    parser.add_argument('--test_data_path', type=str, default='./data/test_data.jsonl', help='测试数据路径')
    parser.add_argument('--pretrained_model_path', type=str, default='./T5_pretrain', help='预训练模型路径')
    parser.add_argument('--batch_size', type=int, default=12, help='批处理大小')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--max_seq_length', type=int, default=512, help='最大序列长度')
    args = parser.parse_args()
    return args