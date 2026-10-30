// frontend/src/types/knowledge.ts
export interface KnowledgeDocItem {
  id: number;
  title: string;
  source: string | null;
  source_type: string | null;
  created_at: string | null;
}

export interface KnowledgeDocListResponse {
  total: number;
  items: KnowledgeDocItem[];
}

export interface KnowledgeUploadResponse {
  doc_id: number;
  title: string;
  chunk_count: number;
  message: string;
}