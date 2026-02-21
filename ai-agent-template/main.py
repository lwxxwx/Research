from dotenv import load_dotenv
from agent.graph import build_agent_graph

load_dotenv(".env.example")

if __name__ == "__main__":
    print("✅ AI 智能体启动：LangChain + LangGraph + Llama + RAG")
    graph = build_agent_graph()

    result = graph.invoke({
        "input": "你好",
        "context": "",
        "response": ""
    })

    print("\n最终输出：", result["response"])
