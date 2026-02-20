from sentence_transformers import SentenceTransformer
import numpy as np

# ====================== 1. 基础功能验证（原有代码） ======================
print("===== 基础模型功能测试 =====")
try:
    # 加载本地模型
    model = SentenceTransformer("/data/models/embeddings/all-MiniLM-L6-v2")
    # 测试基础编码功能
    emb = model.encode("测试成功")
    print(f"基础编码功能正常，向量维度: {emb.shape}")
    print("基础功能测试通过 ✅")
except Exception as e:
    print(f"基础功能测试失败 ❌: {e}")
    exit(1)

# ====================== 2. sentence-transformers[train] 功能测试（修正版） ======================
print("\n===== train 扩展功能测试 =====")
try:
    # 测试train扩展的核心组件：训练参数配置、损失函数、数据加载器
    from sentence_transformers import InputExample, losses
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from torch.utils.data import DataLoader, Dataset
    
    # 1. 定义标准Dataset（而非IterableDataset），支持shuffle
    class SimpleTrainDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            return self.examples[idx]
    
    # 2. 创建测试用的训练样本
    train_examples = [
        InputExample(texts=["句子1", "句子2"], label=1.0),
        InputExample(texts=["句子1", "句子3"], label=0.0)
    ]
    
    # 3. 创建标准数据集和数据加载器（支持shuffle）
    train_dataset = SimpleTrainDataset(train_examples)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
    
    # 4. 初始化损失函数（train扩展核心组件）
    train_loss = losses.CosineSimilarityLoss(model=model)
    
    # 5. 初始化训练参数（train扩展核心配置）
    training_args = SentenceTransformerTrainingArguments(
        output_dir="./tmp_train",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        logging_steps=10,
    )
    
    print("✅ DataLoader (带shuffle) 创建成功")
    print("✅ CosineSimilarityLoss 损失函数初始化成功")
    print("✅ SentenceTransformerTrainingArguments 训练参数初始化成功")
    print("train 扩展功能测试通过 ✅")
except ImportError as e:
    print(f"train 扩展功能测试失败 ❌: 缺少依赖 - {e}")
    print("请执行: pip install 'sentence-transformers[train]'")
except Exception as e:
    print(f"train 扩展功能测试失败 ❌: {e}")

# ====================== 3. sentence-transformers[extras] 功能测试 ======================
print("\n===== extras 扩展功能测试 =====")
try:
    # 测试extras扩展的核心功能：FAISS索引、更多评估指标
    from sentence_transformers import util
    import faiss
    
    # 1. 生成测试向量库
    corpus = ["测试句子1", "测试句子2", "测试句子3"]
    corpus_embeddings = model.encode(corpus)
    
    # 2. 使用FAISS创建索引（extras扩展核心功能）
    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    index.add(np.array(corpus_embeddings))
    
    # 3. 测试语义搜索（util中的高级功能）
    query = "测试句子"
    query_embedding = model.encode(query)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=2)
    
    print(f"✅ FAISS 索引创建成功，索引维度: {index.d}")
    print(f"✅ 语义搜索功能正常，搜索结果数量: {len(hits[0])}")
    print("extras 扩展功能测试通过 ✅")
except ImportError as e:
    print(f"extras 扩展功能测试失败 ❌: 缺少依赖 - {e}")
    print("请执行: pip install 'sentence-transformers[extras]'")
except Exception as e:
    print(f"extras 扩展功能测试失败 ❌: {e}")

print("\n===== 所有测试完成 =====")

