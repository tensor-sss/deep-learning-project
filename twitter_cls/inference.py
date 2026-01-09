import torch
from config import set_args
from model import Model
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_helper import MyDataset, Collater
from transformers import AutoTokenizer


def evaluate(model, test_dataloader):
    model.eval()  
    all_pred_label = []
    for batch in tqdm(test_dataloader):
        if torch.cuda.is_available():
            batch = [t.cuda() for t in batch]
        input_ids, attention_mask, token_type_ids = batch
        # print(feat.size())  # torch.Size([32, 15])
        # print(label.size())   # torch.Size([32])
        out = model(input_ids, attention_mask, token_type_ids)  # 等价 model.forward(feat)
        # print(out.size())   # torch.Size([32, 3])
        _, pred_label = torch.max(out, dim=-1)

        pred_label = pred_label.cpu().detach().numpy().tolist()
        all_pred_label.extend(pred_label)
    return all_pred_label


if __name__ == '__main__':
    args = set_args()
    df = pd.read_csv("./data/test_data.csv")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model, fast = True)

    test_dataset = MyDataset(df, tokenizer, is_train=False)

    collater = Collater(is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collater.collate_fn)

    model = Model(num_label=2)
    model.load_state_dict(torch.load('./output/epoch1_model.bin'))

    if torch.cuda.is_available():
        model.cuda()
    all_pred_label = evaluate(model, test_dataloader)

    temp_df = pd.DataFrame()
    temp_df['id'] = df['id']
    temp_df['target'] = all_pred_label
    temp_df.to_csv("./output/submit_df.csv", index=False)




    