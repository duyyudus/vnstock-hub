import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, List
import json
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class LLMTaskProviderSelection(BaseModel):
    provider: str = Field(..., min_length=1)
    model: str | None = None


def _settings_yaml_path() -> Path:
    configured_path = os.getenv("SETTINGS_YAML_PATH")
    if configured_path:
        return Path(configured_path)
    return Path(__file__).resolve().parents[2] / "settings.yaml"


def _load_settings_yaml_defaults() -> dict[str, Any]:
    path = _settings_yaml_path()
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"settings.yaml at {path} must be a YAML mapping")

    defaults: dict[str, Any] = {"settings_yaml_path": str(path)}
    app_config = data.get("app") or {}
    sync_config = data.get("sync") or {}
    llm_config = data.get("llm") or {}
    auth_config = data.get("auth") or {}

    if isinstance(app_config, dict):
        if "api_v1_prefix" in app_config:
            defaults["api_v1_prefix"] = app_config["api_v1_prefix"]

    if isinstance(sync_config, dict):
        for key in (
            "target_rpm",
            "max_workers",
            "chunk_days",
            "rate_limit_fixed_wait_seconds",
            "rate_limit_max_wait_seconds",
        ):
            if key in sync_config:
                defaults[f"sync_{key}"] = sync_config[key]
    if isinstance(llm_config, dict):
        if "task_config" in llm_config:
            defaults["llm_task_config"] = json.dumps(llm_config["task_config"])
        if "request_timeout_seconds" in llm_config:
            defaults["llm_request_timeout_seconds"] = llm_config["request_timeout_seconds"]

    if isinstance(auth_config, dict):
        if "algorithm" in auth_config:
            defaults["jwt_algorithm"] = auth_config["algorithm"]
        if "access_token_expire_minutes" in auth_config:
            defaults["access_token_expire_minutes"] = auth_config["access_token_expire_minutes"]

    return defaults


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/vnstock_hub"
    
    # API
    api_v1_prefix: str = "/api/v1"
    
    # CORS
    cors_origins: str = '["http://localhost:5173","http://localhost:3000"]'
    
    # Sync behavior
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
    settings_yaml_path: str = str(_settings_yaml_path())

    # Auth/JWT
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Sync admin allowlist (JSON array of lowercased emails)
    sync_admin_emails: str = "[]"

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
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _load_settings_yaml_defaults,
            file_secret_settings,
        )


settings = Settings()
