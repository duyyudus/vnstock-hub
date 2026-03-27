import json

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
