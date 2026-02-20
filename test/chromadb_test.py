import os
import chromadb

# 👇 只加这一句，让 Chroma 读你本地的模型，不再下载
os.environ["CHROMA_CACHE_DIR"] = "/data/models/embeddings"

# 下面完全是你原来的代码，一字没动！
client = chromadb.Client()

coll = client.create_collection(name="test_collection")

coll.add(
    ids=["id1"],
    documents=["这是一条测试文本，用来验证向量库正常工作"]
)

results = coll.query(
    query_texts=["测试"],
    n_results=1
)

print("查询成功！")
print("找到内容：", results["documents"][0][0])

