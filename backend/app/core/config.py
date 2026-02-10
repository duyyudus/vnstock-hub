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
    price_sync_target_rpm: int = 150
    price_sync_max_workers: int = 6
    price_sync_chunk_days: int = 1095
    price_sync_rate_limit_max_retries: int = 12
    price_sync_retry_base_delay_seconds: float = 5.0
    price_sync_retry_max_delay_seconds: float = 60.0

    # LLM providers (OpenAI-compatible) in JSON list format
    llm_providers: str = "[]"
    llm_request_timeout_seconds: int = 30

    # Non-environmental config (YAML)
    settings_yaml_path: str = str(Path(__file__).resolve().parents[2] / "settings.yaml")

    # Auth/JWT
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Sync admin allowlist (JSON array of lowercased emails)
    sync_admin_emails: str = "[]"
    
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
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
