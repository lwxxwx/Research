from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. 定义状态
class State(TypedDict):
    message: str

# 2. 定义一个最简单节点
def hello_node(state: State):
    return {"message": "LangGraph 运行正常！"}

# 3. 建图
builder = StateGraph(State)
builder.add_node("hello", hello_node)
builder.set_entry_point("hello")
builder.add_edge("hello", END)

# 4. 编译运行
graph = builder.compile()
result = graph.invoke({"message": ""})

print("✅ LangGraph 测试通过")
print("结果：", result["message"])
