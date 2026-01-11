"""
@file   : inference_v1.py
@time   : 2026-01-11
"""
from transformers import GPT2LMHeadModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./gpt2_pretrain')

question = '我爱你'
inputs = tokenizer(question, return_tensors='pt')
# {input_ids: [xxx]}

model = GPT2LMHeadModel.from_pretrained('./gpt2_pretrain')
generation_output = model.generate(**inputs)
print(generation_output)  #
'''
tensor([[ 101, 2769, 4263,  872,  102,  100,  100,  100,  100,  100,  100,  100,
          100,  100,  100,  100,  100,  100,  100,  100]])
'''
for idx, sentence in enumerate(generation_output.sequences):
    print('next sentence %d:\n'%idx,
    tokenizer.decode(sentence).split('<|endoftext|>')[0])
    print('*'*40)

