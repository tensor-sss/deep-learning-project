"""
@file   : model.py
@time   : 2025-12-21
"""
from config import set_args
from torch import nn
from transformers import AutoModel

args = set_args()


class Classify(nn.Module):
    def __init__(self, num_label):
        super(Classify, self).__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_label)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        out = self.linear3(x)
        return out


class Model(nn.Module):
    def __init__(self, num_label):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained(args.pretrain_model)  # bert导进来
        #         input_ids: Optional[torch.Tensor] = None,
        #         attention_mask: Optional[torch.Tensor] = None,
        #         token_type_ids: Optional[torch.Tensor] = None,
        hidden_size = self.bert.config.hidden_size
        self.classify = nn.Linear(hidden_size, num_label)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # cls_output
        # last_layer_all_token_output
        # print(out)   # last_hidden_state    # pooler_output
        last_hidden_state = out.last_hidden_state
        cls_output = out.last_hidden_state[:, 0]

        # print(last_hidden_state.size())  # batch_size, max_len, 768
        # print(cls_output.size())   # batch_size, 768

        # 分类:
        # 方法一: 取last_hidden_state这个矩阵的池化结果
        # 方法二：直接用cls_output
        logits = self.classify(cls_output)
        return logits
























