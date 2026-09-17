# RAG Seed Knowledge 种子知识库（Sprint‑0 Phase‑H）
## 目录分工
- `datasheets/`：器件手册关键片段
- `reference_designs/`：官方参考设计片段
- `application_notes/`：应用笔记AN

## Markdown 文件强制规范
每个知识md头部**必须有YAML frontmatter**，ingest脚本会解析这些元字段写入doc_chunks表。

```yaml
---
source_type: datasheet           # 枚举：datasheet / reference_design / application_note
source_title: STC89C55RC datasheet
source_section: Oscillator Circuit 晶振电路设计
part_numbers: ["STC89C55RC"]    # 关联器件型号数组
related_rule_ids: ["MCU_001"]   # 关联规则ID数组
---
文档片段正文……
