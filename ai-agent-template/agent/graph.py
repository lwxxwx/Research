from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    input: str
    context: str
    response: str

def retrieve_node(state: AgentState):
    print("🔍 执行 RAG 检索")
    return {"context": "参考内容来自知识库"}

def llama_node(state: AgentState):
    print("🤖 本地 Llama 生成回答")
    return {"response": f"你好！我是AI智能体，我收到你的问题：{state['input']}"}

def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("llama", llama_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "llama")
    workflow.add_edge("llama", END)
    return workflow.compile()
