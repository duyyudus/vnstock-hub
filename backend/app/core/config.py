import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import json
from pathlib import Path
from pydantic import BaseModel, Field


class LLMTaskProviderSelection(BaseModel):
    provider: str = Field(..., min_length=1)
    model: str | None = None


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
    llm_task_config: str = "{}"
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
    news_search_provider: str = ""
    news_search_api_key: str | None = None
    news_search_base_url: str = "https://api.search.brave.com/res/v1/web/search"
    news_search_timeout_seconds: int = 15

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
    def llm_task_config_map(self) -> dict[str, list[LLMTaskProviderSelection]]:
        """Parse task-specific LLM routing from JSON string."""
        if not self.llm_task_config:
            return {}

        raw_config = json.loads(self.llm_task_config)
        if not isinstance(raw_config, dict):
            raise ValueError("LLM task config must be a JSON object")

        parsed: dict[str, list[LLMTaskProviderSelection]] = {}
        for task_key, selections in raw_config.items():
            normalized_task_key = str(task_key).strip()
            if not normalized_task_key:
                raise ValueError("LLM task config contains an empty task key")
            if not isinstance(selections, list):
                raise ValueError(f"LLM task config for '{normalized_task_key}' must be a list")
            parsed[normalized_task_key] = [
                LLMTaskProviderSelection.model_validate(selection)
                for selection in selections
            ]
        return parsed

    def resolve_llm_providers(self, task_key: str) -> List[dict]:
        """Resolve task-specific provider/model chain with legacy fallback."""
        providers = self.llm_providers_list
        task_config = self.llm_task_config_map
        resolved_task_key = str(task_key).strip()

        selections = task_config.get(resolved_task_key)
        if selections is None:
            selections = task_config.get("default")
        if not selections:
            return providers

        providers_by_name: dict[str, dict] = {}
        for provider in providers:
            provider_name = str(provider.get("name") or "").strip()
            if not provider_name:
                raise ValueError("Each LLM provider must include a non-empty name")
            if provider_name in providers_by_name:
                raise ValueError(f"Duplicate LLM provider name '{provider_name}'")
            providers_by_name[provider_name] = dict(provider)

        resolved: list[dict] = []
        for selection in selections:
            base_provider = providers_by_name.get(selection.provider)
            if base_provider is None:
                raise ValueError(
                    f"LLM task config for '{resolved_task_key or 'default'}' references unknown provider "
                    f"'{selection.provider}'"
                )
            provider_payload = dict(base_provider)
            if selection.model is not None:
                provider_payload["model"] = selection.model
            resolved.append(provider_payload)
        return resolved

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
