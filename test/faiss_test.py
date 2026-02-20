# 导入faiss库，这是验证安装的第一步
import faiss
import numpy as np

# 1. 定义基础参数（维度、数据量）
d = 64  # 向量维度
n_data = 1000  # 样本数据量

# 2. 生成随机测试数据（浮点型数组是FAISS的标准输入格式）
np.random.seed(42)  # 固定随机种子，保证结果可复现
data = np.random.random((n_data, d)).astype('float32')

# 3. 构建最简单的Flat L2索引（暴力搜索，无优化，最易验证）
index = faiss.IndexFlatL2(d)

# 4. 向索引中添加数据
index.add(data)

# 5. 生成一个查询向量并执行搜索
query = np.random.random((1, d)).astype('float32')
k = 5  # 返回最相似的5个结果
distances, indices = index.search(query, k)

# 6. 输出验证结果
print("✅ FAISS-CPU 安装验证成功！")
print(f"搜索到的Top-{k}相似向量索引：{indices[0]}")
print(f"对应的L2距离：{distances[0]}")
