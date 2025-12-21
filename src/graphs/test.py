from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

def llm_node(state: State):
    model = ChatOpenAI(model="gpt-4o")
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
graph = builder.compile()

# 方式 1：流式输出节点更新
print("=== 流式输出节点更新 ===")
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "Hello"}]},
    stream_mode="updates"
):
    print(chunk)

# 方式 2：流式输出完整状态
print("\n=== 流式输出完整状态 ===")
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "Hello"}]},
    stream_mode="values"  # 返回每一步的完整状态
):
    print(f"Latest message: {chunk['messages'][-1].content}")