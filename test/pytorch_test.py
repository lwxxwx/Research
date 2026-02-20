import os
# ==================== 第一步：置顶配置GPU算力适配环境变量（核心！）====================
# 指定RTX 5090的sm_120算力，让PyTorch运行时编译适配内核
# 必须在import torch前设置，否则环境变量不生效
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

# ==================== 原有调试/同步环境变量配置（保留）====================
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 强制CUDA同步执行，便于精准定位报错
os.environ['TORCH_USE_CUDA_DSA'] = '1'     # 启用CUDA设备端断言，提升报错信息准确性

# ==================== 导入所有依赖（torch需在环境变量设置后导入）====================
import numpy as np
import pandas as pd
import torch

# 基础验证：库版本+GPU状态
print(f"NumPy 版本：{np.__version__}")
print(f"Pandas 版本：{pd.__version__}")
print(f"PyTorch 版本：{torch.__version__}")
print(f"GPU 是否可用：{torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_cap = torch.cuda.get_device_capability(0)
    print(f"当前使用 GPU：{gpu_name}")
    print(f"GPU 算力版本：{gpu_cap[0]}.{gpu_cap[1]}")  # 新增：打印算力，验证是否识别sm_120
print("-" * 50)

# 核心协同验证：Pandas→NumPy→PyTorch(GPU)→计算→回传
try:
    # Pandas创建测试数据
    df = pd.DataFrame({"a": [1,2,3,4,5], "b": [10,20,30,40,50]})
    print("原始Pandas数据：\n", df)

    # 数据流转+GPU计算（全链路）
    gpu_tensor = torch.from_numpy(df.values).cuda()  # 转GPU张量
    gpu_result = gpu_tensor * 2  # GPU上执行计算
    df_result = pd.DataFrame(gpu_result.cpu().numpy(), columns=["a*2", "b*2"])  # 结果回传

    print("\nGPU计算后的Pandas结果（数据翻倍）：\n", df_result)
    print("\n✅ 验证通过！Pandas、NumPy、PyTorch 与 GPU 协同工作正常！")

except RuntimeError as e:
    print(f"\n❌ GPU运算错误：{str(e)[:150]}")  # 延长报错截取长度，便于查看完整核心信息
    print("💡 解决方案：确认环境变量TORCH_CUDA_ARCH_LIST=12.0已置顶设置，且在import torch前执行")
except Exception as e:
    print(f"\n❌ 程序运行错误：{str(e)[:150]}")
