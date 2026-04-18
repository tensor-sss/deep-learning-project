"""
@file   : run_zhipu_api.py
@time   : 2026-04-18
"""
# pip install openai
from openai import OpenAI

client = OpenAI(
    api_key="x",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

while True:
    user_input = input("User:")
    completion = client.chat.completions.create(
        model="glm-4.5-air",
        messages=[
            {"role": "system", "content": "你是一个ai助手，可以回答用户的所有问题"},
            {"role": "user", "content": user_input}
        ],
        top_p=0.7,
        temperature=0.9
    )

    print("Bot:", completion.choices[0].message.content)