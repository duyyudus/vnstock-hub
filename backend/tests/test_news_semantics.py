import json

import httpx
import pytest

from app.core import config
from app.services.llm import (
    NEWS_ARTICLE_CLASSIFICATION_TASK,
    NEWS_ARTICLE_SUMMARY_TASK,
    NEWS_BLOCKED_LABEL_COMPILATION_TASK,
)
from app.services.news import semantics


@pytest.fixture(autouse=True)
def reset_news_semantics_provider_state():
    semantics._provider_failure_until.clear()
    semantics._provider_last_success_at.clear()
    config.settings.llm_task_config = "{}"
    yield
    semantics._provider_failure_until.clear()
    semantics._provider_last_success_at.clear()
    config.settings.llm_task_config = "{}"


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
    monkeypatch.setattr(config.settings, "llm_task_config", "{}")

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

    first_payload = await semantics._call_json_llm(NEWS_BLOCKED_LABEL_COMPILATION_TASK, "system", "user")
    assert first_payload == {"labels": ["macro"]}
    assert calls == ["Bearer bad-key", "Bearer good-key"]

    calls.clear()
    second_payload = await semantics._call_json_llm(NEWS_BLOCKED_LABEL_COMPILATION_TASK, "system", "user")
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
    monkeypatch.setattr(config.settings, "llm_task_config", "{}")

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

    first_payload = await semantics._call_json_llm(NEWS_BLOCKED_LABEL_COMPILATION_TASK, "system", "user")
    assert first_payload == {"topics": ["banking"]}
    assert calls == ["Bearer primary-key", "Bearer fallback-key"]

    calls.clear()
    second_payload = await semantics._call_json_llm(NEWS_BLOCKED_LABEL_COMPILATION_TASK, "system", "user")
    assert second_payload == {"topics": ["banking"]}
    assert calls == ["Bearer fallback-key"]


@pytest.mark.asyncio
async def test_summarize_article_requests_original_language(monkeypatch):
    captured: dict[str, str] = {}

    async def _fake_call_json_llm(task_key: str, system_prompt: str, user_prompt: str):
        captured["task_key"] = task_key
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
    assert captured["task_key"] == NEWS_ARTICLE_SUMMARY_TASK
    assert "original language" in captured["user_prompt"]
    assert "Language hint: vi." in captured["user_prompt"]
    assert "70-90 words" in captured["user_prompt"]
    assert "3-5 short complete sentences" in captured["user_prompt"]
    assert "120 words" in captured["user_prompt"]


@pytest.mark.asyncio
async def test_summarize_article_preserves_overlong_summary_when_model_exceeds_hint(monkeypatch):
    raw_summary = "  ".join(
        [
            "This summary intentionally runs well beyond the suggested feed-card target so we can verify the backend keeps the full model output when the content is still useful.",
            "It adds context about regulatory approvals, refinancing pressure, execution risk, shareholder concerns, and expected timing for follow-up disclosures across multiple business units.",
            "The point of this test is to confirm we treat the word range as guidance only rather than silently trimming the response after generation.",
        ]
    )
    expected_summary = " ".join(raw_summary.split())

    async def _fake_call_json_llm(task_key: str, system_prompt: str, user_prompt: str):
        return {"summary": raw_summary}

    monkeypatch.setattr(semantics, "_call_json_llm", _fake_call_json_llm)

    summary = await semantics.summarize_article(
        "Long summary article",
        "Excerpt",
        "Detailed content",
    )

    assert len(summary.split()) > 70
    assert summary == expected_summary


def test_normalize_summary_text_collapses_whitespace():
    summary = "Revenue   improved sharply.\n\nMargins expanded on lower funding costs.\tShares rose after the update."

    normalized = semantics._normalize_summary_text(summary)

    assert normalized == "Revenue improved sharply. Margins expanded on lower funding costs. Shares rose after the update."


def test_normalize_label_transliterates_vietnamese_text():
    assert semantics._normalize_label("phát hành cổ phiếu riêng lẻ") == "phat_hanh_co_phieu_rieng_le"


def test_display_labels_preserve_human_readable_text_and_dedupe():
    labels = semantics._display_labels(
        [
            "  phát hành cổ phiếu riêng lẻ  ",
            "phat hanh co phieu rieng le",
            "tăng vốn điều lệ ngân hàng",
        ]
    )

    assert labels == ["phát hành cổ phiếu riêng lẻ", "tăng vốn điều lệ ngân hàng"]


@pytest.mark.asyncio
async def test_classify_article_uses_classification_task_key(monkeypatch):
    captured: dict[str, str] = {}

    async def _fake_call_json_llm(task_key: str, system_prompt: str, user_prompt: str):
        captured["task_key"] = task_key
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return {
            "topics": ["banking"],
            "tickers": ["VCB"],
            "sectors": ["banking"],
            "importance": "high",
            "sentiment": "positive",
        }

    monkeypatch.setattr(semantics, "_call_json_llm", _fake_call_json_llm)

    payload = await semantics.classify_article(
        "VCB tang truong loi nhuan",
        "Tin ngan",
        "Noi dung chi tiet",
    )

    assert payload["topics"] == ["banking"]
    assert payload["tickers"] == ["VCB"]
    assert payload["raw_payload"]["display_topics"] == ["banking"]
    assert payload["event_type"] == "earnings"
    assert payload["raw_payload"]["event_type"] == "earnings"
    assert captured["task_key"] == NEWS_ARTICLE_CLASSIFICATION_TASK
    assert "Do not guess." in captured["user_prompt"]
    assert "currencies such as USD" in captured["user_prompt"]
    assert "empty tickers array" in captured["user_prompt"]
    assert "\"event_type\"" in captured["user_prompt"]
    assert "\"event_labels\"" in captured["user_prompt"]


@pytest.mark.asyncio
async def test_classify_article_keeps_display_topics_separate_from_normalized_keys(monkeypatch):
    async def _fake_call_json_llm(task_key: str, system_prompt: str, user_prompt: str):
        return {
            "topics": ["phát hành cổ phiếu riêng lẻ", "tăng vốn điều lệ ngân hàng"],
            "tickers": ["BID"],
            "sectors": ["banking"],
            "importance": "high",
            "sentiment": "positive",
        }

    monkeypatch.setattr(semantics, "_call_json_llm", _fake_call_json_llm)

    payload = await semantics.classify_article(
        "BID duoc mua manh",
        "Co phieu ngan hang tang",
        "Noi dung chi tiet",
    )

    assert payload["topics"] == ["phat_hanh_co_phieu_rieng_le", "tang_von_dieu_le_ngan_hang"]
    assert payload["raw_payload"]["display_topics"] == ["phát hành cổ phiếu riêng lẻ", "tăng vốn điều lệ ngân hàng"]


@pytest.mark.asyncio
async def test_classify_article_does_not_fallback_to_regex_tickers_when_llm_returns_empty(monkeypatch):
    async def _fake_call_json_llm(task_key: str, system_prompt: str, user_prompt: str):
        return {
            "topics": ["technology"],
            "tickers": [],
            "sectors": ["technology"],
            "importance": "medium",
            "sentiment": "neutral",
        }

    monkeypatch.setattr(semantics, "_call_json_llm", _fake_call_json_llm)

    payload = await semantics.classify_article(
        "DOANH NGHIEP NHA NUOC - TRU COT TRONG CUOC CANH TRANH CONG NGHE QUOC GIA",
        "Bai viet nhac den VNPT, USD va SASAC nhung khong co ma co phieu.",
        "Noi dung viet hoa co nhieu tu viet tat nhung khong nen tro thanh ticker.",
    )

    assert payload["tickers"] == []


@pytest.mark.asyncio
async def test_classify_article_keeps_tickers_empty_when_llm_unavailable(monkeypatch):
    async def _fake_call_json_llm(task_key: str, system_prompt: str, user_prompt: str):
        return None

    monkeypatch.setattr(semantics, "_call_json_llm", _fake_call_json_llm)

    payload = await semantics.classify_article(
        "DOANH NGHIEP NHA NUOC - TRU COT TRONG CUOC CANH TRANH CONG NGHE QUOC GIA",
        "Bai viet nhac den VNPT, USD va SASAC nhung khong co ma co phieu.",
        "Noi dung viet hoa co nhieu tu viet tat nhung khong nen tro thanh ticker.",
    )

    assert payload["tickers"] == []
    assert payload["topics"]


@pytest.mark.asyncio
async def test_classify_article_sets_heuristic_event_metadata_when_llm_unavailable(monkeypatch):
    async def _fake_call_json_llm(task_key: str, system_prompt: str, user_prompt: str):
        return None

    monkeypatch.setattr(semantics, "_call_json_llm", _fake_call_json_llm)

    payload = await semantics.classify_article(
        "ABC approves cash dividend and record date",
        "Board confirms dividend payout",
        "The company announced a cash dividend and the upcoming record date for shareholders.",
    )

    assert payload["event_type"] == "dividend"
    assert payload["event_labels"] == ["dividend"]
    assert payload["raw_payload"]["event_type"] == "dividend"


@pytest.mark.asyncio
async def test_compile_blocked_labels_uses_blocked_label_task_key(monkeypatch):
    captured: dict[str, str] = {}

    async def _fake_call_json_llm(task_key: str, system_prompt: str, user_prompt: str):
        captured["task_key"] = task_key
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return {"labels": ["banking", "real_estate"]}

    monkeypatch.setattr(semantics, "_call_json_llm", _fake_call_json_llm)

    labels = await semantics.compile_blocked_labels("banking, real estate")

    assert labels == ["banking", "real_estate"]
    assert captured["task_key"] == NEWS_BLOCKED_LABEL_COMPILATION_TASK
