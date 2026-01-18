
import argparse


def set_args():
    parser = argparse.ArgumentParser(description="Qwen3 Fine-tuning")
    parser.add_argument('--pretrained_model_path', type=str, default='./qwen3-0.5b-pretrain', help='Path to the pre-trained model')
    parser.add_argument('--train_data_path', type=str, default='./data/train_data.jsonl', help='Path to the training data')
    parser.add_argument('--test_data_path', type=str, default='./data/test_data.jsonl', help='Path to the testing data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the fine-tuned model')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length for inputs')
    args = parser.parse_args()
    return args