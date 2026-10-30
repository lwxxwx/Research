// frontend/src/api/knowledge.ts
import { request } from "./http";
import type { KnowledgeDocListResponse, KnowledgeUploadResponse } from "@/types/knowledge";

export function listKnowledgeDocs(): Promise<KnowledgeDocListResponse> {
  return request<KnowledgeDocListResponse>("/knowledge-docs");
}

export function uploadKnowledgeDoc(file: File): Promise<KnowledgeUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return request<KnowledgeUploadResponse>(
    "/knowledge-docs/upload",
    {
      method: "POST",
      body: formData,
      // 注意：request 封装里默认写了 Content-Type: application/json
      // 我们需要覆盖它，让浏览器自动生成 multipart boundary
      headers: {},
    },
    60_000 // 上传解析耗时较长，超时设为 60 秒
  );
}