import os
from huggingface_hub import snapshot_download

# 1. 设置国内镜像（仅本次运行有效）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 下载模型（自动创建目录、自动复刻所有文件/子目录）
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",  # 模型名称
    local_dir="/data/models/embeddings/all-MiniLM-L6-v2",  # 你的目标目录
    local_dir_use_symlinks=False,  # 下载真实文件，不是软链接
    ignore_patterns=["*.git*"]  # 忽略无关文件
)

print("✅ 模型下载完成！所有文件已保存到 /data/models/embeddings/all-MiniLM-L6-v2")
