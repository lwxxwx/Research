# 本地化 MiniLM Embedding 接入指南

> 用途：为 `all-MiniLM-L6-v2` 本地化部署做接入准备，同时作为"新增 Embedding 后端"的通用清单。
> 目标：保证接入后 `switch_embedding_backend.py --to minilm` 一条命令可切换，且不破坏已有后端（fake / openai / zhipu）。
> 状态：骨架占位（`minilm.py` 已建，未注册、未迁移）；完整接入待本地化部署方案确定后执行。

---

## 1. all-MiniLM-L6-v2 关键事实

| 项 | 值 |
|---|---|
| 模型全名 | `sentence-transformers/all-MiniLM-L6-v2` |
| 原生维度 | **384** |
| 维度是否 ≤ 2000 | ✅ 满足 pgvector HNSW 对 `vector` 类型上限 |
| 是否支持自定义维度 | ❌ 固定 384 |
| 常见加载方式 | `sentence-transformers` / `onnxruntime` / `llama.cpp` |
| 归一化 | sentence-transformers 默认 L2 normalize，需确认 |
| 模型体积 | 约 80MB（fp32） |

**与现有后端对比**：

| 后端 | 维度 | 切换需迁移 |
|---|---|---|
| `fake` | 1536 | 否（初始 1536） |
| `openai` | 1536 | 否（与 fake 同维） |
| `zhipu` | 1024 | 是（`c1_embedding_dim_1024`） |
| `minilm` | **384** | 是（需新增 `c2_embedding_dim_384`） |

## 2. 当前占位状态

`app/llm/embeddings/minilm.py` 已建骨架：

- 继承 `EmbeddingProvider`
- 声明 `provider_name = "minilm"` / `dimension = 384`
- `__init__` / `embed_documents` / `embed_query` 均抛 `NotImplementedError`
- **未注册**到 `embedding_factory._PROVIDERS`
- **未加入** `BACKEND_DIM`
- **未新增**迁移
- **未加入** `switch_embedding_backend.py` 的 `--to choices`

**当前运行不受影响**：

- `get_embedding_client("minilm")` 在工厂里查不到，直接抛 `ValueError`，不会走到骨架的 `NotImplementedError`
- `alembic upgrade head` 不会触发任何 384 维迁移（迁移链仍止于 `c1_embedding_dim_1024`）
- 切换脚本 `--to minilm` 会被 argparse 的 `choices` 拒绝
- `tests/test_llm_zhipu.py` 28 passed、底座回归 167 passed 继续有效

**为什么这样设计**：

| 动作 | 现在做 | 未来接入时做 |
|---|---|---|
| 骨架类 | ✅ | — |
| 注册到工厂 | ❌ | ✅ |
| `BACKEND_DIM` 加键 | ❌ | ✅ |
| Alembic 迁移 `c2` | ❌ | ✅ |
| `--to choices` 加 `minilm` | ❌ | ✅ |

后 4 件必须**同批完成**，不能提前。原因见第 3 章步骤 4 的"⚠️ 重要"说明：Alembic 迁移链是线性的，提前生成 `c2` 会在 `alembic upgrade head` 时自动执行，把当前 1024 维 DB 改成 384 维，破坏当前闭环。

## 3. 完整接入步骤（5 步）

> 5 步必须**同批完成**。步骤 4（迁移）提前生成会触发非预期迁移，见步骤 4 的"⚠️ 重要"。

### 步骤 1：实现 `MiniLMEmbedding`

文件：`app/llm/embeddings/minilm.py`

确定以下 4 件事后填写实现：

| 待定项 | 选项 | 说明 |
|---|---|---|
| 加载方式 | A. `sentence-transformers`<br>B. `onnxruntime`<br>C. `llama.cpp` | A 最常用但依赖重；B 轻量需转换模型；C 适合 CPU 低内存 |
| 模型文件来源 | A. `data/models/all-MiniLM-L6-v2/` 挂载<br>B. 打进镜像<br>C. 首次运行下载 | A 最灵活；B 镜像变大；C 需外网 |
| 归一化 | A. L2 normalize（sentence-transformers 默认）<br>B. 不归一化 | 影响余弦距离，A 与现有后端一致 |
| 批大小 | 默认 32 / 64 | 影响 ingest 性能 |

实现骨架（以 sentence-transformers 为例）：

```python
from app.llm.embeddings.base import EmbeddingProvider


class MiniLMEmbedding(EmbeddingProvider):
    provider_name = "minilm"
    dimension = 384

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or "/data/models/all-MiniLM-L6-v2"
        self._model = None  # 延迟加载

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        self._ensure_model()
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()
```

### 步骤 2：注册到工厂

文件：`app/rag/embedding_factory.py`

**改动 1：新增 import**

在现有 import 区 `from app.llm.embeddings.zhipu import ZhipuEmbedding` 之后追加：

```python
from app.llm.embeddings.minilm import MiniLMEmbedding
```

**改动 2：`_PROVIDERS` 加键**

```python
_PROVIDERS = {
    "fake": FakeEmbedding,
    "openai": OpenAIEmbedding,
    "zhipu": ZhipuEmbedding,
    "minilm": MiniLMEmbedding,      # ← 新增
}
```

**注意**：
- 只加这两处，不改 `BACKEND_DIM`（步骤 3 做），不改 `assert_backend_matches_db`（无需改）。
- 本步骤单独完成时，`get_embedding_client("minilm")` 能构造出实例，但 `BACKEND_DIM["minilm"]` 还不存在，`assert_backend_matches_db` 会抛 `ValueError("不支持的 RAG_EMBEDDING_BACKEND=minilm")`。这是预期的，**步骤 2 必须与步骤 3 同批完成**。
- 不要在此步骤后跑 `switch_embedding_backend.py --to minilm`（`BACKEND_DIM` 无 `minilm` 键，会 KeyError）。

### 步骤 3：登记维度

文件：`app/rag/embedding_factory.py`

**改动：`BACKEND_DIM` 加键**

```python
BACKEND_DIM = {
    "fake": 1536,
    "openai": 1536,
    "zhipu": 1024,
    "minilm": 384,                  # ← 新增
}
```

**注意**：
- 只加这一处，不改 `_PROVIDERS`（步骤 2 已做），不改 `assert_backend_matches_db`（无需改）。
- 加完后 `BACKEND_DIM["minilm"] == 384`，`assert_backend_matches_db` 对 `minilm` 的维度校验逻辑生效，但 **DB 列维度仍是 1024**，此时跑 `switch_embedding_backend.py --to minilm` 会因 `current(1024) != target(384)` 触发 `downgrade`，而 `c2_embedding_dim_384` 迁移**还不存在**（步骤 4 做），Alembic 会报错。
- **步骤 3 必须与步骤 4、步骤 5 同批完成**，不能单独跑切换脚本。
- 不要在此步骤后跑 `switch_embedding_backend.py --to minilm`。

### 步骤 4：新增 Alembic 迁移

文件：`backend/alembic/versions/c2_embedding_dim_384.py`

```python
"""embedding dim: -> 384 (minilm)

Revision ID: c2_embedding_dim_384
Revises: c1_embedding_dim_1024
Create Date: 2026-10-21

【S1-C 新增】
  本地化 all-MiniLM-L6-v2（384 维）接入时使用。
  本迁移只改 knowledge_chunk.embedding 列 + 重建 HNSW 索引，不建表。
  清空旧向量与重灌由 scripts/switch_embedding_backend.py 负责。
"""
from alembic import op


revision = "c2_embedding_dim_384"
down_revision = "c1_embedding_dim_1024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_knowledge_chunk_embedding")
    op.execute("UPDATE knowledge_chunk SET embedding = NULL")
    op.execute(
        "ALTER TABLE knowledge_chunk "
        "ALTER COLUMN embedding TYPE vector(384) "
        "USING embedding::vector(384)"
    )
    op.execute(
        "CREATE INDEX idx_knowledge_chunk_embedding "
        "ON knowledge_chunk USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_knowledge_chunk_embedding")
    op.execute("UPDATE knowledge_chunk SET embedding = NULL")
    op.execute(
        "ALTER TABLE knowledge_chunk "
        "ALTER COLUMN embedding TYPE vector(1024) "
        "USING embedding::vector(1024)"
    )
    op.execute(
        "CREATE INDEX idx_knowledge_chunk_embedding "
        "ON knowledge_chunk USING hnsw (embedding vector_cosine_ops)"
    )
```

**要求**：
- `upgrade`：`DROP INDEX` → `UPDATE ... SET embedding = NULL` → `ALTER COLUMN ... TYPE vector(384)` → `CREATE INDEX ... USING hnsw`
- `downgrade`：反向，回到 1024
- **用 `op.execute("ALTER TABLE ... TYPE vector(384)")` 原生 SQL**，不用 `op.alter_column(existing_type=...)`，规避 `atttypmod` 偏差和异常旧维度
- `revision = "c2_embedding_dim_384"`，`down_revision = "c1_embedding_dim_1024"`

**⚠️ 重要**：本迁移**必须与步骤 1–3、5 同批完成**，不能提前生成。原因：Alembic 迁移链是线性的，提前生成 `c2` 会在 `alembic upgrade head` 时自动执行，把当前 1024 维 DB 改成 384 维，而 `RAG_EMBEDDING_BACKEND=zhipu` 仍是 1024 维，维度守卫立即报错，知识库需要重灌。**提前生成 = 提前触发迁移 = 破坏当前闭环。**

### 步骤 5：更新切换脚本 `--to` choices

文件：`scripts/switch_embedding_backend.py`

**改动：`--to` 的 `choices` 加 `minilm`**

```python
parser.add_argument("--to", required=True, choices=["fake", "openai", "zhipu", "minilm"])
```

**注意**：
- 只改 `choices` 这一行，不改其他逻辑。
- **不要改成 `sorted(BACKEND_DIM.keys())`**——`BACKEND_DIM` 在 `main()` 内延迟 import，argparse 阶段还未加载，动态生成会报 `NameError`。
- 加完后 `switch_embedding_backend.py --to minilm --yes` 可以被 argparse 接受，进入后续流程：迁移 → 清库 → 改 `models.py` → 重灌 → 冒烟。
- 此时步骤 1–4 必须已完成，否则会在迁移或构造 Provider 时失败。

**5 步同批完成检查清单**：

| # | 步骤 | 完成 |
|---|---|---|
| 1 | `minilm.py` 实现（不再抛 `NotImplementedError`） | ☐ |
| 2 | `_PROVIDERS` 注册 | ☐ |
| 3 | `BACKEND_DIM` 加键 | ☐ |
| 4 | `c2_embedding_dim_384.py` 迁移 | ☐ |
| 5 | `--to choices` 加 `minilm` | ☐ |

5 项全部打勾后再跑：

```powershell
docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
  exec backend uv run python /scripts/switch_embedding_backend.py --to minilm --yes
```

否则会失败或破坏 DB。

## 4. 接入后验证

5 步全部完成后，按以下 4 项验证。任一项失败，回滚到切换前状态（`--to zhipu`）。

### 4.1 单元测试

在 `tests/test_llm_zhipu.py` 加：

```python
def test_embedding_factory_minilm_returns_instance():
    """
    MiniLM 当前是骨架占位：
    - 未注册到 _PROVIDERS 时，get_embedding_client("minilm") 抛 ValueError
    - 注册后但 __init__ 抛 NotImplementedError 时，pytest.skip
    - 完整接入后，正常返回实例

    三种状态都兼容，不会让 CI 因"尚未接入"而失败。
    """
    from app.llm.embeddings.minilm import MiniLMEmbedding
    from app.rag.embedding_factory import BACKEND_DIM, get_embedding_client

    try:
        client = get_embedding_client("minilm")
    except ValueError:
        pytest.skip("minilm 尚未注册到 _PROVIDERS（骨架占位阶段）")
    except NotImplementedError:
        pytest.skip("minilm 已注册但尚未实现（骨架占位阶段）")

    assert isinstance(client, MiniLMEmbedding)
    assert client.dimension == BACKEND_DIM["minilm"] == 384
```

跑：

```powershell
docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
  exec backend uv run pytest tests/test_llm_zhipu.py -v -k "minilm"
```

### 4.2 切换冒烟

```powershell
docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
  exec backend uv run python /scripts/switch_embedding_backend.py --to minilm --yes
```

预期 5 步全过：迁移 → 清库 → 改 `models.py` → 重灌 → 冒烟。

### 4.3 维度确认

```powershell
docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
  exec backend uv run python /scripts/_diag_embedding.py
```

预期：

```text
embedding type: vector(384)
alembic_version: ['c2_embedding_dim_384']
```

### 4.4 检索冒烟

```powershell
docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
  exec backend uv run pytest tests/test_rag_seed.py -v -k "test_retriever_basic_query"
```

### 4.5 失败回滚

任一项失败，切回 zhipu：

```powershell
docker compose -f "$PWD/infra/docker/docker-compose.yml" -f "$PWD/infra/docker/docker-compose.dev.yml" `
  exec backend uv run python /scripts/switch_embedding_backend.py --to zhipu --yes
```

回滚依赖 `c2` 迁移的 `downgrade`（384 → 1024），已包含在步骤 4 的迁移代码中。

## 5. 通用接入清单（未来新增其他后端复用）

本节把第 3 节的 5 步抽象为通用清单。未来新增任何 Embedding 后端（如 `bge`、`local_xxx`），照此执行。

### 5.1 步骤表

| # | 动作 | 文件 |
|---|---|---|
| 1 | 新建 Provider 类，声明 `provider_name` / `dimension` | `app/llm/embeddings/<name>.py` |
| 2 | 注册到 `_PROVIDERS` | `app/rag/embedding_factory.py` |
| 3 | 登记 `BACKEND_DIM` | `app/rag/embedding_factory.py` |
| 4 | 新增 Alembic 迁移（维度与已有后端不同时） | `backend/alembic/versions/cN_embedding_dim_<dim>.py` |
| 5 | 更新切换脚本 `--to` choices | `scripts/switch_embedding_backend.py` |
| 6 | 加单元测试 | `tests/test_llm_zhipu.py` |
| 7 | 切换冒烟 + 检索冒烟 | 命令行 |

### 5.2 维度冲突检查

**若新后端维度与已有后端相同，跳过步骤 4（迁移）。**

| 场景 | 是否需新迁移 |
|---|---|
| 新增后端维度 = 1536（如另一个 OpenAI 兼容模型） | ❌ 复用初始 1536，无需迁移 |
| 新增后端维度 = 1024（如另一个 1024 维模型） | ❌ 复用 `c1_embedding_dim_1024` |
| 新增后端维度 = 384（如 MiniLM） | ✅ 新增 `c2_embedding_dim_384` |
| 新增后端维度 = 768（如 BGE-base） | ✅ 新增 `c3_embedding_dim_768` |

**判断方法**：查 `BACKEND_DIM` 是否已有该维度值。有则复用，无则新迁移。

### 5.3 同批完成原则

步骤 1–5 **必须同批完成**，不能分批。原因：

- 只做步骤 1–3：`get_embedding_client` 能构造实例，但 `--to` choices 不接受，无法切换；若手动绕过调用，`BACKEND_DIM` 有键但 DB 维度不对，守卫会拦。
- 只做步骤 1–4：迁移生成后 `alembic upgrade head` 自动执行，把当前 DB 改成新维度，但 `RAG_EMBEDDING_BACKEND` 仍是旧后端，守卫立即报错。
- 只做步骤 5：`--to choices` 接受新后端，但 `_PROVIDERS` / `BACKEND_DIM` / 迁移都不存在，脚本中途失败。

**结论**：5 步是一个原子操作，代码、配置、迁移、脚本一起改，一起提交。

### 5.4 提交信息模板

```
feat(embedding): add <name> backend (<dim>-dim)

- app/llm/embeddings/<name>.py: provider implementation
- app/rag/embedding_factory.py: register in _PROVIDERS + BACKEND_DIM
- backend/alembic/versions/cN_embedding_dim_<dim>.py: migration (if dim differs)
- scripts/switch_embedding_backend.py: add to --to choices
- tests/test_llm_zhipu.py: factory dispatch test
```

## 6. 常见坑

| 坑 | 说明 | 规避 |
|---|---|---|
| 维度 > 2000 | `vector` 类型 HNSW 上限 2000，超过建索引失败 | 用 `halfvec`（上限 4000）或降维（Matryoshka） |
| `atttypmod - 4` 偏差 | 部分 pgvector 版本算出 1020 而非 1024 | 读维度用 `format_type(atttypid, atttypmod)` |
| 维度本地硬编码 | 多个事实来源，改一处漏一处 | `BACKEND_DIM` 只在 `embedding_factory` 定义一处 |
| 迁移 `existing_type` 不匹配 | 当前列维度异常时 `op.alter_column` 失败 | 用原生 `ALTER TABLE ... TYPE vector(N)` |
| 测试替身缺 `provider_name` | `ingest_single_file` 从实例读 `provider_name` / `model` | 自定义替身补齐两属性 |
| 无参 `get_embedding_client()` 依赖环境变量 | 测试断言漂移（期望 1536，实际随 `.env` 变） | 测试显式传后端名 |
| `--to` choices 硬编码 | 加新后端忘记改，argparse 拒绝 | 步骤 5 必做 |
| **提前生成迁移** | Alembic 线性链，提前生成会触发非预期迁移 | 步骤 1–5 同批完成 |
| **动态生成 `--to choices`** | `BACKEND_DIM` 在 `main()` 内延迟 import，argparse 阶段未加载 | 硬编码列表，不要 `sorted(BACKEND_DIM.keys())` |
| **`models.py` 未同步** | 迁移改了 DB，`models.py` 仍是旧 `Vector(N)` | 切换脚本 `_sync_models_py` 自动改；手动切换需自己改 |
| **多后端维度混用** | 同一 DB 内不同后端向量维度不一致，检索报错 | 单列 + 迁移 + 切换脚本，不双列并存 |
| **`minilm` 半接入** | 只注册未实现，`__init__` 抛 `NotImplementedError` 但被其他路径调到 | 5 步同批完成，不单独注册 |

## 7. 相关文档

| 文档 | 作用 |
|---|---|
| `app/llm/embeddings/base.py` | `EmbeddingProvider` 契约（`embed_documents` / `embed_query` / `provider_name` / `dimension`） |
| `app/llm/embeddings/minilm.py` | MiniLM 骨架（本文档对应实现） |
| `app/rag/embedding_factory.py` | 工厂 + `BACKEND_DIM` + `assert_backend_matches_db` |
| `scripts/switch_embedding_backend.py` | 切换脚本（迁移 + 清库 + 改 `models.py` + 重灌 + 冒烟） |
| `scripts/_diag_embedding.py` | 维度诊断脚本（本地工具，已加 `.gitignore`） |
| `docs/sprint1/sprint1_changelog.md` | 阶段 C 变更记录（含 5 个坑的详细说明） |
| `backend/alembic/versions/c1_embedding_dim_1024.py` | 智谱 1024 维迁移（`c2` 的参考模板） |

---

## 8. 变更记录

| 日期 | 变更 | 说明 |
|---|---|---|
| 2026-10-21 | 初版 | 本地化 MiniLM 接入指南 + 通用接入清单 |