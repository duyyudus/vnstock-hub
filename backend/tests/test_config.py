import json
import textwrap

import pytest

from app.core import config
from app.services.llm import POSITION_IMAGE_EXTRACTION_TASK


def test_resolve_llm_providers_applies_task_specific_model_override(monkeypatch):
    monkeypatch.setattr(
        config.settings,
        "llm_providers",
        json.dumps(
            [
                {
                    "name": "vision",
                    "base_url": "https://vision.example.com/v1",
                    "api_key": "vision-key",
                    "model": "vision-default",
                },
                {
                    "name": "cheap",
                    "base_url": "https://cheap.example.com/v1",
                    "api_key": "cheap-key",
                    "model": "cheap-default",
                },
            ]
        ),
    )
    monkeypatch.setattr(
        config.settings,
        "llm_task_config",
        json.dumps(
            {
                POSITION_IMAGE_EXTRACTION_TASK: [
                    {"provider": "vision", "model": "vision-fast"},
                    {"provider": "cheap"},
                ]
            }
        ),
    )

    resolved = config.settings.resolve_llm_providers(POSITION_IMAGE_EXTRACTION_TASK)

    assert resolved == [
        {
            "name": "vision",
            "base_url": "https://vision.example.com/v1",
            "api_key": "vision-key",
            "model": "vision-fast",
        },
        {
            "name": "cheap",
            "base_url": "https://cheap.example.com/v1",
            "api_key": "cheap-key",
            "model": "cheap-default",
        },
    ]


def test_resolve_llm_providers_uses_default_task_chain(monkeypatch):
    monkeypatch.setattr(
        config.settings,
        "llm_providers",
        json.dumps(
            [
                {
                    "name": "primary",
                    "base_url": "https://primary.example.com/v1",
                    "api_key": "primary-key",
                    "model": "primary-model",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        config.settings,
        "llm_task_config",
        json.dumps({"default": [{"provider": "primary", "model": "summary-model"}]}),
    )

    resolved = config.settings.resolve_llm_providers("missing_task")

    assert resolved == [
        {
            "name": "primary",
            "base_url": "https://primary.example.com/v1",
            "api_key": "primary-key",
            "model": "summary-model",
        }
    ]


def test_resolve_llm_providers_falls_back_to_legacy_list_without_task_config(monkeypatch):
    providers = [
        {
            "name": "legacy",
            "base_url": "https://legacy.example.com/v1",
            "api_key": "legacy-key",
            "model": "legacy-model",
        }
    ]
    monkeypatch.setattr(config.settings, "llm_providers", json.dumps(providers))
    monkeypatch.setattr(config.settings, "llm_task_config", "{}")

    resolved = config.settings.resolve_llm_providers("unconfigured_task")

    assert resolved == providers


def test_resolve_llm_providers_rejects_unknown_provider_reference(monkeypatch):
    monkeypatch.setattr(
        config.settings,
        "llm_providers",
        json.dumps(
            [
                {
                    "name": "primary",
                    "base_url": "https://primary.example.com/v1",
                    "api_key": "primary-key",
                    "model": "primary-model",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        config.settings,
        "llm_task_config",
        json.dumps({POSITION_IMAGE_EXTRACTION_TASK: [{"provider": "missing"}]}),
    )

    with pytest.raises(ValueError, match="unknown provider"):
        config.settings.resolve_llm_providers(POSITION_IMAGE_EXTRACTION_TASK)


def test_settings_load_non_secret_defaults_from_yaml(tmp_path, monkeypatch):
    settings_yaml = tmp_path / "settings.yaml"
    settings_yaml.write_text(
        textwrap.dedent(
            """
            app:
              api_v1_prefix: /custom/api
            auth:
              algorithm: HS512
              access_token_expire_minutes: 15
            sync:
              target_rpm: 42
              max_workers: 3
              chunk_days: 90
              rate_limit_fixed_wait_seconds: 7
              rate_limit_max_wait_seconds: 77
            llm:
              request_timeout_seconds: 12
              task_config:
                default:
                  - provider: primary
            brokers: []
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SETTINGS_YAML_PATH", str(settings_yaml))
    for env_key in (
        "API_V1_PREFIX",
        "JWT_ALGORITHM",
        "ACCESS_TOKEN_EXPIRE_MINUTES",
        "SYNC_TARGET_RPM",
        "SYNC_MAX_WORKERS",
        "SYNC_CHUNK_DAYS",
        "SYNC_RATE_LIMIT_FIXED_WAIT_SECONDS",
        "SYNC_RATE_LIMIT_MAX_WAIT_SECONDS",
        "LLM_REQUEST_TIMEOUT_SECONDS",
        "LLM_TASK_CONFIG",
    ):
        monkeypatch.delenv(env_key, raising=False)

    loaded = config.Settings(_env_file=None)

    assert loaded.api_v1_prefix == "/custom/api"
    assert loaded.jwt_algorithm == "HS512"
    assert loaded.access_token_expire_minutes == 15
    assert loaded.sync_target_rpm == 42
    assert loaded.sync_max_workers == 3
    assert loaded.sync_chunk_days == 90
    assert loaded.sync_rate_limit_fixed_wait_seconds == 7
    assert loaded.sync_rate_limit_max_wait_seconds == 77
    assert loaded.llm_request_timeout_seconds == 12
    assert loaded.llm_task_config_map["default"][0].provider == "primary"


def test_env_overrides_settings_yaml_defaults(tmp_path, monkeypatch):
    settings_yaml = tmp_path / "settings.yaml"
    settings_yaml.write_text(
        textwrap.dedent(
            """
            app:
              api_v1_prefix: /yaml/api
            sync:
              target_rpm: 42
            brokers: []
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SETTINGS_YAML_PATH", str(settings_yaml))
    monkeypatch.setenv("API_V1_PREFIX", "/env/api")
    monkeypatch.setenv("SYNC_TARGET_RPM", "99")

    loaded = config.Settings(_env_file=None)

    assert loaded.api_v1_prefix == "/env/api"
    assert loaded.sync_target_rpm == 99
