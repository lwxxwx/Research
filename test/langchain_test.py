# 测试：LangChain + Skill + 链 全套正常
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import FakeListLLM

@tool
def add(a: int, b: int):
    """加法工具"""
    return a + b

# 假LLM（不调用任何模型）
llm = FakeListLLM(responses=["工具调用正常"])

prompt = ChatPromptTemplate.from_template("计算 {a} + {b}")
chain = prompt | llm

if __name__ == "__main__":
    res = chain.invoke({"a": 10, "b": 20})
    print(res)
    print("✅ LangChain + Tool 整套环境正常！")

