-- ==============================================
-- 002_schema.sql Sprint0 DB Schema v1.2‑enhance
-- Source of Truth：禁止代码自动建表，只能执行本SQL脚本
--
-- 【Sprint‑0约束说明】
-- review_defect：Sprint‑0建表并实现JSON解析写入；Sprint‑0不使用该表做业务查询；
-- review_result.review_output_json 作为权威备份存储；Sprint‑1全面切换到review_defect做查询统计。
-- tasks / benchmark_results：Sprint‑1新增，不在Sprint0 schema。
--
-- P0必选表(10张)：
-- 1. users
-- 2. project
-- 3. schematic_case
-- 4. ir_document
-- 5. rule_definition
-- 6. rule_execution
-- 7. review_result
-- 8. feedback_item
-- 9. knowledge_doc
-- 10. knowledge_chunk
--
-- P1可选表：
-- 11. rule_candidates（规则演进闭环，Sprint‑0建表，Sprint‑1业务使用）
-- 12. review_defect（缺陷明细表，Sprint‑0建表基础写入，Sprint‑1完善查询）
-- ==============================================
-- 通用：updated_at 自动更新触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- ==============================================
-- 1. users
-- ==============================================
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(128) NOT NULL UNIQUE,
    email VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==============================================
-- 2. project
-- ==============================================
CREATE TABLE IF NOT EXISTS project (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    description TEXT,
    owner_id BIGINT REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE TRIGGER update_project_updated_at BEFORE UPDATE ON project FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==============================================
-- 3. schematic_case 黄金用例/测试案例
-- ==============================================
CREATE TABLE IF NOT EXISTS schematic_case (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT REFERENCES project(id) ON DELETE CASCADE,
    case_id VARCHAR(64) NOT NULL,         -- case001
    case_type VARCHAR(32) NOT NULL,         -- A / B
    case_path VARCHAR(512),                     -- data/cases/case001
    expert_calibration_md TEXT,                 -- expert_reasoning.md 内容
    expected_review_json JSONB,                 -- expected_review.json
    evaluation_yaml TEXT,                       -- evaluation.yaml 内容
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(project_id, case_id)
);
CREATE TRIGGER update_schematic_case_updated_at BEFORE UPDATE ON schematic_case FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==============================================
-- 4. ir_document 原理图IR中间表示存储
-- ==============================================
CREATE TABLE IF NOT EXISTS ir_document (
    id BIGSERIAL PRIMARY KEY,
    schematic_case_id BIGINT REFERENCES schematic_case(id) ON DELETE CASCADE,
    ir_json JSONB NOT NULL,
    ir_schema_version VARCHAR(32) NOT NULL,  -- v1.0
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==============================================
-- 5. rule_definition 规则定义（对应yaml规则）
-- ==============================================
CREATE TABLE IF NOT EXISTS rule_definition (
    id BIGSERIAL PRIMARY KEY,
    rule_id VARCHAR(64) NOT NULL UNIQUE,     -- POWER_001
    rule_name VARCHAR(256) NOT NULL,
    rule_category VARCHAR(64) NOT NULL,      -- POWER / CLOCK / IFACE
    rule_yaml TEXT NOT NULL,
    description TEXT,
    severity VARCHAR(20) DEFAULT 'medium',
    enabled BOOLEAN DEFAULT TRUE,
    version VARCHAR(20) DEFAULT 'v1.0',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE TRIGGER update_rule_definition_updated_at BEFORE UPDATE ON rule_definition FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==============================================
-- 6. rule_execution 单次规则执行记录
-- ==============================================
CREATE TABLE IF NOT EXISTS rule_execution (
    id BIGSERIAL PRIMARY KEY,
    ir_document_id BIGINT REFERENCES ir_document(id) ON DELETE CASCADE,
    rule_def_id BIGINT REFERENCES rule_definition(id),
    hit BOOLEAN NOT NULL,
    evidence_json JSONB,
    execution_log TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==============================================
-- 7. review_result 评审结果总输出（Sprint0：Rule‑Only）
-- ==============================================
CREATE TABLE IF NOT EXISTS review_result (
    id BIGSERIAL PRIMARY KEY,
    ir_document_id BIGINT REFERENCES ir_document(id) ON DELETE CASCADE,
    task_id VARCHAR(64),
    review_output_json JSONB NOT NULL,         -- 完整报告JSON（向后兼容）
    is_rule_only BOOLEAN NOT NULL DEFAULT TRUE,

    -- V1.2顶层摘要字段（从 review_output_json 提取，便于查询）
    category VARCHAR(64) NOT NULL DEFAULT 'other',
    risk VARCHAR(16) NOT NULL DEFAULT 'medium',
    review_status VARCHAR(32) NOT NULL DEFAULT 'AI_CONFIRMED',

    -- V1.2 三项指标
    evidence_coverage FLOAT,                     -- EVC
    ai_new_effective_rate FLOAT,                 -- AI_NER
    expert_adoption_rate FLOAT,                  -- EAR

    -- 计数统计
    total_defects INTEGER DEFAULT 0,
    ai_confirmed_count INTEGER DEFAULT 0,
    need_expert_review_count INTEGER DEFAULT 0,
    low_confidence_count INTEGER DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_review_result_review_status ON review_result(review_status);
CREATE INDEX IF NOT EXISTS idx_review_result_category ON review_result(category);
CREATE INDEX IF NOT EXISTS idx_review_result_task_id ON review_result(task_id);

-- ==============================================
-- 8. review_defect 缺陷明细表（新增，Sprint‑0建表，Sprint‑1完善查询）
-- 支持：逐条缺陷查询 / 反馈精准关联 / 按review_status过滤
-- ==============================================
CREATE TABLE IF NOT EXISTS review_defect (
    id BIGSERIAL PRIMARY KEY,
    review_result_id BIGINT REFERENCES review_result(id) ON DELETE CASCADE,
    defect_id VARCHAR(64) NOT NULL UNIQUE,   -- DEF‑20260807‑001

    -- V1.2 10字段规范（完全拆解）
    category VARCHAR(64) NOT NULL DEFAULT 'other',
    location JSONB NOT NULL,                 -- {sheet, path, coords}
    component VARCHAR(100),
    net VARCHAR(100),
    risk VARCHAR(16) NOT NULL DEFAULT 'medium',
    evidence JSONB NOT NULL DEFAULT '[]',    -- [{source, section, reason}]
    root_cause TEXT NOT NULL,
    suggestion TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.5,

    -- 质量状态（三态）
    review_status VARCHAR(32) NOT NULL DEFAULT 'AI_CONFIRMED',

    -- 来源追踪
    origin VARCHAR(32) NOT NULL DEFAULT 'rule',  -- rule / ai_discovered / hybrid
    rule_id VARCHAR(64),                         -- 关联 rule_definition.rule_id

    -- 反馈聚合（便于快速查询，避免每次统计扫feedback表）
    feedback_type VARCHAR(64),                   -- 最新/最严重反馈类型
    feedback_count INTEGER DEFAULT 0,
    latest_feedback_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_review_defect_review_result ON review_defect(review_result_id);
CREATE INDEX IF NOT EXISTS idx_review_defect_defect_id ON review_defect(defect_id);
CREATE INDEX IF NOT EXISTS idx_review_defect_review_status ON review_defect(review_status);
CREATE INDEX IF NOT EXISTS idx_review_defect_category ON review_defect(category);
CREATE INDEX IF NOT EXISTS idx_review_defect_origin ON review_defect(origin);
CREATE INDEX IF NOT EXISTS idx_review_defect_rule_id ON review_defect(rule_id);

-- ==============================================
-- 9. feedback_item 6类反馈【Sprint‑0 Phase‑I 完整V1.2字段】
-- ==============================================
CREATE TABLE IF NOT EXISTS feedback_item (
    id BIGSERIAL PRIMARY KEY,
    review_result_id BIGINT REFERENCES review_result(id) ON DELETE CASCADE,
    review_defect_id BIGINT REFERENCES review_defect(id) ON DELETE CASCADE,  -- 【新增】精准关联到缺陷
    feedback_type VARCHAR(64) NOT NULL,

    -- 原始SQL存量字段
    comment TEXT,
    expert_suggestion TEXT,
    suggestion_adopted VARCHAR(16),              -- yes / partial / no

    -- V1.2 Phase‑I扩展字段
    suggestion_diff_json JSONB NOT NULL DEFAULT '[]',
    rule_candidate_ref VARCHAR(64),
    attached_refs JSONB DEFAULT '[]',

    created_by BIGINT REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 数据库层强制6种反馈类型，字面量严格对齐 app/schemas/feedback.py FeedbackType枚举
ALTER TABLE feedback_item DROP CONSTRAINT IF EXISTS check_feedback_item_type;
ALTER TABLE feedback_item ADD CONSTRAINT check_feedback_item_type CHECK (
    feedback_type IN (
        'correct_defect',
        'false_positive',
        'false_negative',
        'suggestion_update',
        'new_rule_candidate',
        'knowledge_gap'
    )
);

CREATE INDEX IF NOT EXISTS idx_feedback_item_type ON feedback_item(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_item_result ON feedback_item(review_result_id);
CREATE INDEX IF NOT EXISTS idx_feedback_item_defect ON feedback_item(review_defect_id);

-- ==============================================
-- 10. knowledge_doc 原始知识库文档
-- ==============================================
CREATE TABLE IF NOT EXISTS knowledge_doc (
    id BIGSERIAL PRIMARY KEY,
    title VARCHAR(512) NOT NULL,
    source VARCHAR(256),
    source_type VARCHAR(32),                     -- datasheet / reference_design / review_case
    content_md TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    version VARCHAR(20) DEFAULT 'v1.0',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE TRIGGER update_knowledge_doc_updated_at BEFORE UPDATE ON knowledge_doc FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==============================================
-- 11. knowledge_chunk 向量化分片
-- ==============================================
CREATE TABLE IF NOT EXISTS knowledge_chunk (
    id BIGSERIAL PRIMARY KEY,
    knowledge_doc_id BIGINT REFERENCES knowledge_doc(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(1536),                       -- 适配OpenAI embedding
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_embedding ON knowledge_chunk USING hnsw (embedding vector_cosine_ops);

-- ==============================================
-- 12. rule_candidates 规则演进候选（P1，Sprint‑0建表，Sprint‑1业务深度使用）
-- 【重要说明】task_id：Sprint0仅普通字段，**无外键；Sprint‑1 tasks表创建后再补外键约束**
-- ==============================================
CREATE TABLE IF NOT EXISTS rule_candidates (
    id BIGSERIAL PRIMARY KEY,
    candidate_id VARCHAR(64) NOT NULL UNIQUE,   -- RC‑20260807‑001
    from_feedback_id BIGINT REFERENCES feedback_item(id),
    case_id VARCHAR(64),
    task_id BIGINT, -- Sprint0：普通字段，无外键；Sprint‑1 tasks表创建后再补外键约束
    title VARCHAR(256) NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(16),
    evidence_refs JSONB NOT NULL DEFAULT '[]',
    proposed_yaml TEXT,
    status VARCHAR(32) NOT NULL DEFAULT 'proposed',  -- proposed/designing/benchmarked/accepted/rejected
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TRIGGER update_rule_candidates_updated_at BEFORE UPDATE ON rule_candidates FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- rule_candidates全套索引（对齐test_db_schema.py冒烟测试）
CREATE INDEX IF NOT EXISTS idx_rule_candidates_status ON rule_candidates(status);
CREATE INDEX IF NOT EXISTS idx_rule_candidates_case_id ON rule_candidates(case_id);
CREATE INDEX IF NOT EXISTS idx_rule_candidates_from_feedback_id ON rule_candidates(from_feedback_id);
