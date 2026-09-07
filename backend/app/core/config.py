# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List, Optional

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
        if "localhost" in v or "127.0.0.1" in v:
            raise ValueError("DATABASE_URL 在容器内部不能使用 localhost/127.0.0.1，请使用服务名 postgres")
        return v

settings = Settings()