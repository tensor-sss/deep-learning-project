
from transformers import AutoModelForCausalLM, AutoTokenizer
# CausalLM 自回归式   只用了transforemr的decoder部分

model_name = "./output/model_epoch_0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "写首诗"
messages = [
    {"role": "user", "content": prompt},
]

# 把用户输入的结构化数据 包装成一个特定的字符串
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
print(text)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print(model_inputs)  # input_ids, attention_mask

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
