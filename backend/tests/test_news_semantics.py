import json

import httpx
import pytest

from app.core import config
from app.services.news import semantics


@pytest.fixture(autouse=True)
def reset_news_semantics_provider_state():
    semantics._provider_failure_until.clear()
    semantics._provider_last_success_at.clear()
    yield
    semantics._provider_failure_until.clear()
    semantics._provider_last_success_at.clear()


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict, url: str):
        self.status_code = status_code
        self._payload = payload
        self._request = httpx.Request("POST", url)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self._request,
                response=httpx.Response(self.status_code, request=self._request),
            )

    def json(self) -> dict:
        return self._payload


@pytest.mark.asyncio
async def test_call_json_llm_uses_successful_fallback_after_provider_failure(monkeypatch):
    monkeypatch.setattr(
        config.settings,
        "llm_providers",
        json.dumps(
            [
                {
                    "name": "broken-primary",
                    "base_url": "https://primary.example.com",
                    "api_key": "bad-key",
                    "model": "model-a",
                },
                {
                    "name": "healthy-fallback",
                    "base_url": "https://fallback.example.com",
                    "api_key": "good-key",
                    "model": "model-b",
                },
            ]
        ),
    )

    calls: list[str] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, *, json: dict, headers: dict):
            authorization = headers["Authorization"]
            calls.append(authorization)
            if authorization == "Bearer bad-key":
                return _FakeResponse(status_code=400, payload={"error": "bad request"}, url=url)
            return _FakeResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": '{"labels":["macro"]}'}}]},
                url=url,
            )

    monkeypatch.setattr(semantics.httpx, "AsyncClient", FakeAsyncClient)

    first_payload = await semantics._call_json_llm("system", "user")
    assert first_payload == {"labels": ["macro"]}
    assert calls == ["Bearer bad-key", "Bearer good-key"]

    calls.clear()
    second_payload = await semantics._call_json_llm("system", "user")
    assert second_payload == {"labels": ["macro"]}
    assert calls == ["Bearer good-key"]


@pytest.mark.asyncio
async def test_call_json_llm_falls_back_when_primary_returns_empty_choices(monkeypatch):
    monkeypatch.setattr(
        config.settings,
        "llm_providers",
        json.dumps(
            [
                {
                    "name": "empty-primary",
                    "base_url": "https://primary.example.com",
                    "api_key": "primary-key",
                    "model": "model-a",
                },
                {
                    "name": "healthy-fallback",
                    "base_url": "https://fallback.example.com",
                    "api_key": "fallback-key",
                    "model": "model-b",
                },
            ]
        ),
    )

    calls: list[str] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, *, json: dict, headers: dict):
            authorization = headers["Authorization"]
            calls.append(authorization)
            if authorization == "Bearer primary-key":
                return _FakeResponse(status_code=200, payload={"choices": []}, url=url)
            return _FakeResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": '{"topics":["banking"]}'}}]},
                url=url,
            )

    monkeypatch.setattr(semantics.httpx, "AsyncClient", FakeAsyncClient)

    first_payload = await semantics._call_json_llm("system", "user")
    assert first_payload == {"topics": ["banking"]}
    assert calls == ["Bearer primary-key", "Bearer fallback-key"]

    calls.clear()
    second_payload = await semantics._call_json_llm("system", "user")
    assert second_payload == {"topics": ["banking"]}
    assert calls == ["Bearer fallback-key"]


@pytest.mark.asyncio
async def test_summarize_article_requests_original_language(monkeypatch):
    captured: dict[str, str] = {}

    async def _fake_call_json_llm(system_prompt: str, user_prompt: str):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return {"summary": "Tom tat tieng Viet"}

    monkeypatch.setattr(semantics, "_call_json_llm", _fake_call_json_llm)

    summary = await semantics.summarize_article(
        "Co phieu ABC tang manh",
        "Doanh thu tang",
        "Noi dung bai viet bang tieng Viet",
        language="vi",
    )

    assert summary == "Tom tat tieng Viet"
    assert "original language" in captured["user_prompt"]
    assert "Language hint: vi." in captured["user_prompt"]
