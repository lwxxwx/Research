from langchain_community.llms import LlamaCpp
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import Dict
import re
import atexit  # 解决析构报错的核心库

# ====================== 全局变量 & 析构修复 ======================
llm = None  # 全局模型实例，用于手动释放

def clean_up_llm():
    """手动释放模型资源，解决析构报错"""
    global llm
    if llm is not None:
        try:
            llm.close()  # 手动关闭模型
        except:
            pass
    llm = None

# 注册退出钩子：程序结束时自动调用清理函数
atexit.register(clean_up_llm)

# ====================== 1. 初始化本地 GPU 模型 ======================
def init_local_llm():
    """初始化本地 llama.cpp 模型（GPU 加速）"""
    global llm
    llm = LlamaCpp(
        model_path="/home/hzxx/Downloads/qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        n_ctx=4096,                     # 上下文窗口大小
        n_gpu_layers=-1,                # 全部层放 GPU（核心！）
        n_threads=8,                    # CPU 线程数（根据你的电脑调整）
        temperature=0.0,                # 0温度，输出绝对稳定
        verbose=False,                  # 关闭冗余日志
        n_batch=512,                    # 增大批次，避免n_batch警告
    )
    return llm

# ====================== 2. 定义工具（Skill） ======================
# 工具1：计算器（数学计算）
def calculator_tool(expression: str) -> str:
    """
    数学计算器工具
    参数：expression - 合法的数学表达式（仅支持+、-、*、/、()）
    """
    try:
        # 严格过滤表达式，只保留数字和合法运算符（修复正则）
        safe_expr = re.sub(r'[^0-9+\-*/().]', '', expression)
        if not safe_expr:
            return f"计算失败：表达式为空或包含非法字符（原始输入：{expression}）"
        
        result = eval(safe_expr, {"__builtins__": None}, {})
        return f"""
✅ 计算器工具调用成功
├─ 原始表达式：{expression}
├─ 过滤后表达式：{safe_expr}
└─ 计算结果：{safe_expr} = {result}
"""
    except SyntaxError:
        return f"计算失败：表达式语法错误（{expression}），请检查格式（如括号是否配对）"
    except ZeroDivisionError:
        return f"计算失败：表达式包含除以0（{expression}）"
    except Exception as e:
        return f"计算失败：{str(e)}（原始输入：{expression}）"

# 工具2：模拟天气查询（可替换为真实 API）
def weather_tool(city: str) -> str:
    """
    天气查询工具
    参数：city - 城市名称（中文）
    """
    # 模拟返回天气数据（实际可对接高德/百度天气 API）
    weather_data = {
        "北京": "晴，20-28℃，微风，空气质量优",
        "上海": "多云，22-30℃，南风3级，空气质量良",
        "广州": "雷阵雨，25-32℃，东风2级，空气质量良",
        "深圳": "阴，24-31℃，北风1级，空气质量优",
        "杭州": "小雨，21-27℃，西风2级，空气质量优"
    }
    city_clean = re.sub(r'[^一二三四五六七八九十百千万亿零北京上海广州深圳杭州]', '', city)
    weather = weather_data.get(city_clean, f"暂无{city}的天气数据（支持查询：北京/上海/广州/深圳/杭州）")
    
    return f"""
✅ 天气查询工具调用成功
├─ 查询城市：{city}
├─ 清洗后城市名：{city_clean}
└─ 天气结果：{weather}
"""

# 封装工具列表
def get_tools():
    """获取所有可用工具（带详细描述）"""
    tools = [
        Tool(
            name="计算器",
            func=calculator_tool,
            description="仅用于解决数学计算问题，输入必须是合法的数学表达式（如'100+200*3'、'(50-20)/5'）"
        ),
        Tool(
            name="天气查询",
            func=weather_tool,
            description="仅用于查询城市天气，输入必须是中文城市名称（如'北京'、'上海'）"
        )
    ]
    return tools

# ====================== 3. 定义 LangGraph 状态 ======================
class AgentState(Dict):
    """智能体状态：包含问题、思考、工具调用结果、最终答案"""
    question: str
    thought: str
    tool_result: str
    answer: str

# ====================== 4. 思考节点（核心修复：严格解析） ======================
def think_node(state: AgentState) -> AgentState:
    """思考：判断问题是否需要调用工具，严格解析输出格式"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个严格遵守格式的工具调用助手，必须按以下规则输出：
1. 只输出一行内容，无任何多余文字、注释、示例、解释
2. 需要调用工具：输出「工具名|参数」（如：计算器|100+200*3、天气查询|北京）
3. 不需要调用工具：输出「直接回答|答案」（如：直接回答|人工智能是模拟人类智能的技术）
4. 参数必须简洁，不能包含多余描述"""),
        ("user", "问题：{question}")
    ])
    
    # 调用本地模型思考
    chain = prompt | llm
    raw_thought = chain.invoke({"question": state["question"]}).strip()
    
    # 深度清洗思考结果（解决模型输出冗余的核心）
    # 步骤1：移除所有无关前缀（如"思考结果："、"Assistant："等）
    clean_thought = re.sub(r'^(思考结果：|Assistant：|回答：|：)', '', raw_thought)
    # 步骤2：只保留第一行（避免模型输出多行）
    clean_thought = clean_thought.split("\n")[0].strip()
    # 步骤3：移除特殊字符（修复正则报错的核心！）
    clean_thought = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9+\-*/().|]', '', clean_thought)
    
    state["thought"] = clean_thought
    return state

# ====================== 5. 工具调用节点（增强错误处理） ======================
def tool_node(state: AgentState) -> AgentState:
    """执行工具调用（带详细错误处理）"""
    thought = state["thought"]
    state["tool_result"] = ""  # 初始化
    
    # 校验格式
    if "|" not in thought:
        state["tool_result"] = f"❌ 工具调用失败：思考结果格式错误（无分隔符|），原始输入：{thought}"
        return state
    
    # 拆分工具名和参数（只拆分一次，避免参数含|）
    parts = thought.split("|", 1)
    if len(parts) < 2:
        state["tool_result"] = f"❌ 工具调用失败：参数为空，原始输入：{thought}"
        return state
    
    tool_name = parts[0].strip()
    tool_args = parts[1].strip()
    
    # 空值校验
    if not tool_name:
        state["tool_result"] = f"❌ 工具调用失败：工具名为空，原始输入：{thought}"
        return state
    if not tool_args:
        state["tool_result"] = f"❌ 工具调用失败：工具参数为空，原始输入：{thought}"
        return state
    
    # 查找并调用工具
    tools = {t.name: t for t in get_tools()}
    if tool_name in tools:
        try:
            state["tool_result"] = tools[tool_name].func(tool_args)
        except Exception as e:
            state["tool_result"] = f"❌ 工具执行异常：{str(e)}（工具名：{tool_name}，参数：{tool_args}）"
    else:
        state["tool_result"] = f"❌ 未找到工具：{tool_name}（可用工具：{list(tools.keys())}）"
    
    return state

# ====================== 6. 总结节点（生成友好最终回答） ======================
def summarize_node(state: AgentState) -> AgentState:
    """总结工具结果，生成详细、友好的最终回答"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的智能助手，根据以下信息生成详细、友好的最终回答：
1. 如果有工具调用结果，先说明工具调用情况，再给出最终结论
2. 如果没有工具调用结果，直接回答用户问题
3. 回答要简洁明了，结构清晰"""),
        ("user", """
问题：{question}
思考过程：{thought}
工具调用结果：{tool_result}
请生成最终回答：""")
    ])
    
    chain = prompt | llm
    answer = chain.invoke({
        "question": state["question"],
        "thought": state["thought"],
        "tool_result": state["tool_result"]
    }).strip()
    
    # 清洗最终回答（移除冗余内容）
    answer = re.sub(r'^\n+|\n+$', '', answer)
    state["answer"] = answer
    return state

# ====================== 7. 分支判断（路由逻辑） ======================
def should_call_tool(state: AgentState) -> str:
    """判断下一步是调用工具还是直接总结"""
    thought = state["thought"]
    
    # 直接回答逻辑
    if thought.startswith("直接回答|"):
        # 提取直接回答的内容
        answer_parts = thought.split("|", 1)
        if len(answer_parts) > 1:
            state["answer"] = answer_parts[1].strip()
        else:
            state["answer"] = "无法回答该问题"
        return END
    
    # 工具调用逻辑
    elif "|" in thought:
        return "tool_node"
    
    # 无法判断，直接总结
    else:
        state["tool_result"] = f"⚠️ 无法识别思考结果，直接回答（原始输入：{thought}）"
        return "summarize_node"

# ====================== 8. 构建 LangGraph 工作流 ======================
def build_agent_graph():
    """构建智能体工作流"""
    # 创建状态图
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("think_node", think_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("summarize_node", summarize_node)
    
    # 设置入口点
    graph.set_entry_point("think_node")
    
    # 添加条件边（分支逻辑）
    graph.add_conditional_edges(
        "think_node",
        should_call_tool,
        {
            "tool_node": "tool_node",
            "summarize_node": "summarize_node",
            END: END
        }
    )
    
    # 工具调用后总结
    graph.add_edge("tool_node", "summarize_node")
    
    # 总结后结束
    graph.add_edge("summarize_node", END)
    
    # 编译图
    return graph.compile()

# ====================== 9. 主函数：运行智能体 ======================
if __name__ == "__main__":
    # 初始化模型（核心！）
    print("="*50)
    print("正在初始化本地 GPU 模型（RTX 5090）...")
    llm = init_local_llm()
    print("✅ 模型初始化完成！")
    print("="*50 + "\n")
    
    # 构建工作流
    agent_graph = build_agent_graph()
    
    # 测试案例（覆盖正常/异常场景）
    test_cases = [
        "100加上200乘以3等于多少？",
        "北京今天的天气怎么样？",
        "人工智能是什么？",
        "100/0等于多少？",  # 异常案例：除以0
        "纽约今天的天气？"   # 异常案例：无数据城市
    ]
    
    # 运行智能体
    for i, question in enumerate(test_cases, 1):
        print(f"\n【测试案例 {i}】")
        print(f"📌 用户问题：{question}")
        
        # 执行工作流
        result = agent_graph.invoke({
            "question": question,
            "thought": "",
            "tool_result": "",
            "answer": ""
        })
        
        # 详细输出结果
        print(f"🤔 思考过程：{result['thought']}")
        if result["tool_result"]:
            print(f"🔧 工具调用结果：{result['tool_result']}")
        print(f"💡 最终回答：{result['answer']}")
        print("-"*50)
    
    # 手动清理模型（解决析构报错）
    clean_up_llm()
    print("\n✅ 所有测试完成，资源已正常释放！")
