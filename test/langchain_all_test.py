"""
一站式验证 LangChain 全家桶功能
包含：langchain/core/community + LangGraph + Tool/Skill
纯本地运行，无外部依赖
"""
# ==================== 1. 基础库导入测试 ====================
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool
    from langchain_community.llms import FakeListLLM
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    print("✅ 所有核心库导入成功！")
except ImportError as e:
    print(f"❌ 库导入失败: {e}")
    exit(1)

# ==================== 2. LangChain 基础功能测试 ====================
def test_langchain_basic():
    print("\n--- 测试 LangChain 基础功能 ---")
    try:
        # 测试 Prompt 模板
        prompt = ChatPromptTemplate.from_messages([("user", "你好")])
        # 测试 Fake LLM
        llm = FakeListLLM(responses=["基础功能正常！"])
        # 测试链调用
        chain = prompt | llm
        result = chain.invoke({})
        print(f"✅ 基础链调用成功: {result}")
        return True
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return False

# ==================== 3. Tool/Skill 功能测试 ====================
def test_langchain_tool():
    print("\n--- 测试 Tool/Skill 功能 ---")
    try:
        # 定义简单工具
        @tool
        def add(a: int, b: int) -> int:
            """加法工具"""
            return a + b
        
        # 调用工具
        res = add.invoke({"a": 2, "b": 3})
        assert res == 5, "加法工具结果错误"
        print(f"✅ Tool/Skill 调用成功: 2+3={res}")
        return True
    except Exception as e:
        print(f"❌ Tool/Skill 测试失败: {e}")
        return False

# ==================== 4. LangGraph 功能测试 ====================
def test_langgraph():
    print("\n--- 测试 LangGraph 功能 ---")
    try:
        # 定义状态
        class State(TypedDict):
            message: str
        
        # 定义节点
        def hello_node(state: State):
            return {"message": "LangGraph 运行正常！"}
        
        # 构建并运行图
        builder = StateGraph(State)
        builder.add_node("hello", hello_node)
        builder.set_entry_point("hello")
        builder.add_edge("hello", END)
        graph = builder.compile()
        result = graph.invoke({"message": ""})
        
        print(f"✅ LangGraph 运行成功: {result['message']}")
        return True
    except Exception as e:
        print(f"❌ LangGraph 测试失败: {e}")
        return False

# ==================== 5. 总入口 ====================
if __name__ == "__main__":
    print("===== 开始全量测试 LangChain 全家桶 =====")
    
    # 依次执行所有测试
    test1 = test_langchain_basic()
    test2 = test_langchain_tool()
    test3 = test_langgraph()
    
    # 最终结果汇总
    print("\n===== 测试结果汇总 =====")
    if all([test1, test2, test3]):
        print("🎉 所有功能测试通过！LangChain 环境完全正常！")
    else:
        print("❌ 部分功能测试失败，请检查依赖版本！")
