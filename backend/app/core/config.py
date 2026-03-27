import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import json
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/vnstock_hub"
    
    # API
    api_v1_prefix: str = "/api/v1"
    
    # CORS
    cors_origins: str = '["http://localhost:5173","http://localhost:3000"]'
    
    # vnstock API
    vnstock_api_key: str | None = None
    use_vnstock_alt: bool = False
    use_vnstock_data_alt: bool = False
    sync_target_rpm: int = 150
    sync_max_workers: int = 6
    sync_chunk_days: int = 365
    sync_rate_limit_fixed_wait_seconds: float = 30.0
    sync_rate_limit_max_wait_seconds: float = 1200.0

    # LLM providers (OpenAI-compatible) in JSON list format
    llm_providers: str = "[]"
    llm_request_timeout_seconds: int = 30

    # Non-environmental config (YAML)
    settings_yaml_path: str = str(Path(__file__).resolve().parents[2] / "settings.yaml")
    news_sources_yaml_path: str = str(Path(__file__).resolve().parents[2] / "news_sources.yaml")

    # Auth/JWT
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Sync admin allowlist (JSON array of lowercased emails)
    sync_admin_emails: str = "[]"

    # News ingestion
    news_ingestion_enabled: bool = True
    news_poll_interval_seconds: float = 120.0
    news_default_poll_interval_minutes: int = 30
    news_ingestion_batch_size: int = 5

    # Build number (set via BUILD_NUMBER env var in CI/Docker; falls back to git hash)
    build_number: str | None = None
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from JSON string."""
        return json.loads(self.cors_origins)

    @property
    def llm_providers_list(self) -> List[dict]:
        """Parse LLM providers from JSON string."""
        if not self.llm_providers:
            return []
        return json.loads(self.llm_providers)

    @property
    def sync_admin_emails_list(self) -> List[str]:
        """Parse sync admin emails from JSON string."""
        if not self.sync_admin_emails:
            return []
        raw = json.loads(self.sync_admin_emails)
        return [str(email).strip().lower() for email in raw if str(email).strip()]
    
    model_config = SettingsConfigDict(
        env_file=os.getenv("APP_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
    )


settings = Settings()
