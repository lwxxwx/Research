import torch
import time

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("✅ CUDA 可用，使用 GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("❌ CUDA 不可用，使用 CPU")
    device = torch.device("cpu")

# 创建两个大矩阵并移到GPU
a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

print("开始矩阵乘法...")
start_time = time.time()

# 执行矩阵乘法，这会占用大量GPU算力
c = a @ b

end_time = time.time()
print(f"✅ 矩阵乘法完成，耗时: {end_time - start_time:.2f} 秒")

# 保持程序运行，方便观察nvitop
print("程序将在10秒后退出...")
time.sleep(10)
