"""
@file   : run_agent.py
@time   : 2026-04-18
"""
import datetime
from typing import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode


class MyState(TypedDict):
    messages: Annotated[list, add_messages]

# messages: [{}, {}, {}]

'''
@tool作用:
把函数转成一种格式化的数据 让大模型容易理解
{
    "name": "get_time",
    "description": "获取当前时间", 
    "parameters": {
    }
    ...
}
'''

@tool
def get_time() -> str:
    """
    获取当前时间

    Returns:
        str: 当前时间
    """
    cur_time = datetime.datetime.today()
    return cur_time.strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气

    Args:
        city: 城市名称

    Returns:
        str: 天气信息
    """
    d = {
        "北京": "晴天, 温度10-20摄氏度, 微风",
        "西安": "小雨, 温度15-25摄氏度, 大风",
        "上海": "晴天, 温度20-30摄氏度, 无风"
    }

    if city in d:
        return d[city]
    else:
        return "抱歉，暂时无法获取{}的天气信息".format(city)



tools = [get_time, get_weather]


def create_llm():
    llm = ChatOpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="x",
        model="glm-4.5-air",
        temperature=0.9
    )
    return llm.bind_tools(tools)
llm = create_llm()   # 最终绑定了工具的大模型


# 节点1: 问大模型
def call_model(state):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


# 条件边
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return "end"


def build_workflow():
    workflow = StateGraph(MyState)

    # 加节点
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))   # 注意 要把之前的所有工具打包放到这块

    # 加关系(加边)      一定要以START开始   要以END结尾
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


if __name__ == '__main__':
    workflow = build_workflow()

    # user_input = {"messages": [HumanMessage(content="今天是几号")]}
    user_input = {"messages": [HumanMessage(content="天津天气怎么样")]}
    res = workflow.invoke(user_input)
    for msg in res['messages']:
        print(msg)
        print("*"*100)


