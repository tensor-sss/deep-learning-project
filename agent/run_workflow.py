"""
@file   : run_workflow.py
@time   : 2026-04-18
"""
# 干一件事的步骤
# 让用户输入一个童话故事的标题  -> 大模型生成童话故事 ->  生成关于这个故事一些哲理
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END



llm = ChatOpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    api_key="x",
    model="glm-4.5-air",
    temperature=0.9
)


class MyState(TypedDict):
    user_query: str
    story: str
    story_analysis: str


# 节点1: 生成童话故事
def generate_story(state):
    print("走到第一个节点: 开始生成故事ing")
    user_query = state['user_query']
    messages = [
        SystemMessage(content='你是一个儿童故事大王，可以根据用户给定的题目生成一篇300字童话故事，风格跟安徒生故事类似。'),
        HumanMessage(content=user_query)
    ]
    result = llm.invoke(messages)
    story_content = result.content
    return {"story": story_content}


# 节点2: 分析童话故事
def analyze_story(state):
    print("走到第二个节点: 开始分析故事ing")
    story = state['story']
    messages = [
        SystemMessage(content='你是一个儿童教育专家，可以根据用户给定的童话故事内容分析出所蕴含的哲理，以小朋友口吻回答。'),
        HumanMessage(content=story)
    ]
    result = llm.invoke(messages)
    analysis_res = result.content
    return {"story_analysis": analysis_res}


def build_workflow():
    workflow = StateGraph(MyState)

    # 加节点
    workflow.add_node("generate_story", generate_story)
    workflow.add_node("analyze_story", analyze_story)

    # 加关系(加边)      一定要以START开始   要以END结尾
    workflow.add_edge(START, "generate_story")
    workflow.add_edge("generate_story", "analyze_story")
    workflow.add_edge("analyze_story", END)
    return workflow.compile()


if __name__ == '__main__':
    workflow = build_workflow()
    user_input = "写一个乌龟和蚂蚁的故事"
    res = workflow.invoke({"user_query": user_input})
    print(res)
