"""
@file   : run_zhipu_api_v2.py
@time   : 2026-04-18
"""
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    api_key="x",
    model="glm-4.5-air",
    temperature=0.9
)

res = llm.invoke("你是谁")
print(res)



