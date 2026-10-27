// frontend/src/types/evidence.ts
// 对齐核对文档第 6 章（Evidence）

/** 证据类型（Pydantic Literal） */
export type EvidenceType = "rag_ref" | "ir_ref";

/**
 * 单条证据（对齐 app/schemas/report.py::Evidence）。
 *
 * 条件必填语义（TS 无法直接表达，运行时按 type 判断）：
 *   - type="rag_ref"：source / section / reason 必有值
 *   - type="ir_ref" ：detail 必有值
 *   - source_type 可选
 */
export interface Evidence {
  type: EvidenceType;
  // rag_ref 专用
  source?: string | null;
  section?: string | null;
  reason?: string | null;
  // ir_ref 专用
  detail?: string | null;
  // 可选扩展
  source_type?: string | null;
}