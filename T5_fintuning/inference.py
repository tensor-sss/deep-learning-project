
from config import set_args


def get_model(args, device):
    # config = T5Config.from_pretrained(args.pretrained_model_path)
    # model = T5ForConditionalGeneration(config=config)   # 不加载预训练权重

    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_path)  # 加载预训练权重
    model = model.to(device)
    return model


if __name__ == '__main__':
    args = set_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_path = './output/xxx'
    args.pretrained_model_path = save_path

    model = get_model(args, device)

    input_text = "上班族可以炒股吗？"
    result = model.generate(input_text)
    print(result)  # tensor([[   0, 32128,  3293,  4638,  4638,  4638,    3]])
    output_text = tokenizer.decode(result[0], skip_special_tokens=True)
    print(output_text)  

