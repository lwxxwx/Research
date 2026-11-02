# app/core/config.py
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="/app/.env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    # ========== 豆包风格：核心必需字段 ==========
    app_env: str
    secret_key: str
    database_url: str
    database_echo: bool = False
    log_level: str = "INFO"
    data_dir: str
    llm_provider: str
    llm_api_key: str

    # ========== 新增：支持CORS ==========
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # ========== 新增：支持LLM扩展 ==========
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    # ========== 智谱 AI（LLM_PROVIDER=zhipu 时使用） ==========
    zhipuai_api_key: Optional[str] = None
    zhipuai_model: str = "glm-4-flash"
    zhipuai_embedding_model: str = "embedding-3"
    # -------- 【✅新增】Embedding配置 --------
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_api_key: Optional[str] = None
    #rag_embedding_backend: str = Field(default="fake", env="RAG_EMBEDDING_BACKEND")
    #rag_embedding_backend: str = Field(
    #default="fake",
    #json_schema_extra={"env": "RAG_EMBEDDING_BACKEND"}
    #)
    # ⚠️ V1.3：Pydantic v2 已废弃 v1 的 env= 写法；字段名默认映射
    # 环境变量 RAG_EMBEDDING_BACKEND（大小写不敏感）
    rag_embedding_backend: str = "fake"


    @property
    def is_dev(self) -> bool:
        return self.app_env.lower() == "dev"

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, values):
        env = values.data.get("app_env", "dev")
        if env.lower() == "production" and v.startswith("please_replace_this"):
            raise ValueError("生产环境必须修改 SECRET_KEY，禁止使用默认占位值")
        return v

    @field_validator("database_url")
    @classmethod
    def check_db_host(cls, v: str):
        # 允许 CI（APP_ENV=test）和本地开发（APP_ENV=dev）使用 localhost
        # 只对生产环境（prod）强制使用服务名 postgres
        import os
        env = os.environ.get("APP_ENV", "dev").lower()
        if env == "prod" and ("localhost" in v or "127.0.0.1" in v):
            raise ValueError("DATABASE_URL 在生产环境不能使用 localhost/127.0.0.1，请使用服务名 postgres")
        return v

settings = Settings()
