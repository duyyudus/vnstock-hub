from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
import unicodedata

import httpx

from app.core.config import settings
from app.services.llm import (
    NEWS_ARTICLE_CLASSIFICATION_TASK,
    NEWS_ARTICLE_DISCUSSION_TASK,
    NEWS_ARTICLE_DISCUSSION_DECISION_TASK,
    NEWS_ARTICLE_DISCUSSION_QUERY_TASK,
    NEWS_ARTICLE_SUMMARY_TASK,
    NEWS_QUICK_GLANCE_TASK,
    NEWS_BLOCKED_LABEL_COMPILATION_TASK,
)
from app.services.llm.llm_client import _build_chat_url, _extract_json_payload


logger = logging.getLogger("vnstock_hub.news")
PROVIDER_FAILURE_COOLDOWN_SECONDS = 300.0
PROVIDER_CONFIGURATION_COOLDOWN_SECONDS = 1800.0
ARTICLE_SUMMARY_TARGET_MIN_WORDS = 70
ARTICLE_SUMMARY_TARGET_MAX_WORDS = 90
ARTICLE_SUMMARY_SOFT_MAX_WORDS = 120
QUICK_GLANCE_SUMMARY_SOFT_MAX_WORDS = 180
QUICK_GLANCE_HIGHLIGHT_SOFT_MAX_WORDS = 80
_provider_failure_until: dict[str, float] = {}
_provider_last_success_at: dict[str, float] = {}
PLACEHOLDER_PROVIDER_VALUES = {
    "",
    "your_key",
    "your-api-key",
    "your_api_key",
    "your_key_here",
    "your-api-key-here",
    "your_api_key_here",
    "replace_me",
    "changeme",
    "change_me",
}

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings": ("profit", "earnings", "revenue", "guidance", "quarter", "financial results"),
    "dividend": ("dividend", "cash payout", "stock dividend", "record date"),
    "banking": ("bank", "banking", "credit growth", "npl", "deposit", "loan"),
    "real_estate": ("real estate", "property", "housing", "apartment", "land"),
    "regulation": ("government", "ministry", "circular", "decree", "policy", "regulation"),
    "macro": ("inflation", "gdp", "cpi", "interest rate", "exchange rate", "macro"),
    "commodities": ("oil", "gold", "gas", "commodity", "coal", "steel"),
    "technology": ("technology", "software", "ai", "data center", "chip", "semiconductor"),
}

EVENT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings": (
        "earnings",
        "quarterly earnings",
        "financial results",
        "profit",
        "loi nhuan",
        "lợi nhuận",
        "revenue",
        "doanh thu",
        "quarter",
        "annual report",
        "q1",
        "q2",
        "q3",
        "q4",
    ),
    "dividend": (
        "dividend",
        "cash payout",
        "stock dividend",
        "record date",
        "ex-dividend",
        "bonus shares",
        "co tuc",
        "cổ tức",
        "ngay dang ky cuoi cung",
        "ngày đăng ký cuối cùng",
    ),
    "capital_raise": (
        "capital raise",
        "private placement",
        "rights issue",
        "share issuance",
        "bond issuance",
        "increase charter capital",
        "fundraising",
    ),
    "insider_trading": (
        "insider trading",
        "internal trading",
        "major shareholder",
        "registered to buy",
        "registered to sell",
        "bought shares",
        "sold shares",
    ),
    "management_change": (
        "ceo",
        "cfo",
        "chairman",
        "appoint",
        "appointment",
        "resignation",
        "dismissal",
        "leadership",
        "management change",
    ),
    "regulatory": (
        "regulator",
        "investigation",
        "sanction",
        "inspection",
        "compliance",
        "decree",
        "circular",
        "policy",
        "regulation",
        "license",
        "approval",
    ),
    "mna": (
        "merger",
        "acquisition",
        "takeover",
        "buyout",
        "strategic stake",
        "strategic investment",
        "sell stake",
        "divestment",
    ),
    "analyst_view": (
        "target price",
        "broker report",
        "research report",
        "recommendation",
        "upgrade",
        "downgrade",
        "outperform",
        "underperform",
        "buy rating",
        "sell rating",
    ),
    "macro_policy": (
        "interest rate",
        "gdp",
        "cpi",
        "inflation",
        "exchange rate",
        "central bank",
        "monetary policy",
        "fiscal policy",
        "trade policy",
    ),
}

EVENT_TYPE_DISPLAY_LABELS: dict[str, str] = {
    "earnings": "earnings",
    "dividend": "dividend",
    "capital_raise": "capital raise",
    "insider_trading": "insider trading",
    "management_change": "management change",
    "regulatory": "regulatory",
    "mna": "M&A",
    "analyst_view": "analyst view",
    "macro_policy": "macro policy",
    "other": "other",
}

SECTOR_MAP = {
    "banking": "banking",
    "real_estate": "real_estate",
    "technology": "technology",
    "commodities": "materials",
}

POSITIVE_KEYWORDS = ("beat", "growth", "record high", "profit rises", "upgrade", "expansion")
NEGATIVE_KEYWORDS = ("loss", "decline", "cut", "warning", "downgrade", "investigation")
HIGH_IMPORTANCE_KEYWORDS = (
    "merger",
    "acquisition",
    "earnings",
    "dividend",
    "profit warning",
    "default",
    "regulation",
)
BLOCKED_SPLIT_PATTERN = re.compile(r"[\n,;|]+")


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_label(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", _ascii_fold(value).strip().lower())
    return cleaned.strip("_")


def _normalize_labels(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        cleaned = _normalize_label(value)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _normalize_display_label(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _display_labels(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _normalize_display_label(str(value))
        if not cleaned:
            continue
        dedupe_key = _normalize_label(cleaned) or cleaned.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(cleaned)
    return normalized


def _normalize_summary_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_discussion_text(text: str, *, limit: int) -> str:
    normalized_lines: list[str] = []
    previous_blank = False
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        normalized_line = re.sub(r"[^\S\n]+", " ", raw_line).strip()
        if not normalized_line:
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(normalized_line)
        previous_blank = False

    normalized = "\n".join(normalized_lines).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 1)].rstrip()}…"


def _normalize_discussion_query(text: str, *, limit: int = 240) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 1)].rstrip()}…"


def _provider_key(provider: dict[str, Any]) -> str:
    name = str(provider.get("name") or "").strip()
    base_url = str(provider.get("base_url") or "").strip()
    model = str(provider.get("model") or "").strip()
    return f"{name}|{base_url}|{model}"


def _ordered_providers(providers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = time.monotonic()

    def sort_key(provider: dict[str, Any]) -> tuple[int, float, str]:
        key = _provider_key(provider)
        cooling_down = _provider_failure_until.get(key, 0.0) > now
        last_success = _provider_last_success_at.get(key, 0.0)
        return (1 if cooling_down else 0, -last_success, key)

    ordered = sorted(providers, key=sort_key)
    if ordered and all(_provider_failure_until.get(_provider_key(item), 0.0) > now for item in ordered):
        return sorted(
            ordered,
            key=lambda provider: (_provider_failure_until.get(_provider_key(provider), 0.0), -_provider_last_success_at.get(_provider_key(provider), 0.0)),
        )
    return ordered


def _provider_credential_value(provider: dict[str, Any]) -> str:
    return str(provider.get("api_key") or "").strip()


def _provider_has_placeholder_credentials(provider: dict[str, Any]) -> bool:
    api_key = _provider_credential_value(provider)
    if not api_key:
        return True
    normalized = re.sub(r"[^a-z0-9]+", "_", api_key.lower()).strip("_")
    return normalized in PLACEHOLDER_PROVIDER_VALUES


def _mark_provider_success(provider_key: str) -> None:
    _provider_failure_until.pop(provider_key, None)
    _provider_last_success_at[provider_key] = time.monotonic()


def _mark_provider_failure(provider_key: str, *, cooldown_seconds: float) -> None:
    _provider_failure_until[provider_key] = time.monotonic() + cooldown_seconds


async def _call_json_llm(task_key: str, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
    try:
        providers = settings.resolve_llm_providers(task_key)
    except Exception as exc:
        logger.warning("news_llm_config_error task=%s error=%s", task_key, exc)
        return None
    if not providers:
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for provider in _ordered_providers(providers):
        provider_key = _provider_key(provider)
        provider_name = str(provider.get("name") or provider_key)
        if _provider_has_placeholder_credentials(provider):
            _mark_provider_failure(
                provider_key,
                cooldown_seconds=PROVIDER_CONFIGURATION_COOLDOWN_SECONDS,
            )
            logger.warning(
                "news_llm_provider_skipped provider=%s reason=missing_or_placeholder_credentials",
                provider_name,
            )
            continue
        try:
            url = _build_chat_url(str(provider["base_url"]))
            payload = {
                "model": provider["model"],
                "messages": messages,
                "temperature": 0.1,
            }
            headers = {
                "Authorization": f"Bearer {_provider_credential_value(provider)}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=settings.llm_request_timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise ValueError("Provider returned no choices")
            content = choices[0].get("message", {}).get("content")
            if not content:
                raise ValueError("Provider returned empty content")
            extracted_payload = _extract_json_payload(content)
            if not isinstance(extracted_payload, dict):
                raise ValueError("Provider returned a non-object JSON payload")
            _mark_provider_success(provider_key)
            return extracted_payload
        except httpx.HTTPStatusError as exc:
            cooldown_seconds = (
                PROVIDER_CONFIGURATION_COOLDOWN_SECONDS
                if 400 <= exc.response.status_code < 500
                else PROVIDER_FAILURE_COOLDOWN_SECONDS
            )
            _mark_provider_failure(provider_key, cooldown_seconds=cooldown_seconds)
            logger.warning("news_llm_failure provider=%s error=%s", provider_name, exc)
        except Exception as exc:
            _mark_provider_failure(
                provider_key,
                cooldown_seconds=PROVIDER_FAILURE_COOLDOWN_SECONDS,
            )
            logger.warning("news_llm_failure provider=%s error=%s", provider_name, exc)
            continue
    return None


def _heuristic_topics(text: str) -> list[str]:
    lowered = text.lower()
    topics = [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    return _normalize_labels(topics)

def _heuristic_sentiment(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in NEGATIVE_KEYWORDS):
        return "negative"
    if any(keyword in lowered for keyword in POSITIVE_KEYWORDS):
        return "positive"
    return "neutral"


def _heuristic_importance(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in HIGH_IMPORTANCE_KEYWORDS):
        return "high"
    if len(lowered) > 250:
        return "medium"
    return "low"


def _heuristic_event_payload(text: str) -> dict[str, Any]:
    lowered = text.lower()
    matched_types = [
        event_type
        for event_type, keywords in EVENT_TYPE_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    event_type = matched_types[0] if matched_types else "other"
    event_labels = [
        EVENT_TYPE_DISPLAY_LABELS.get(item, item.replace("_", " "))
        for item in matched_types[:3]
    ] or [EVENT_TYPE_DISPLAY_LABELS[event_type]]
    return {
        "event_type": event_type,
        "event_labels": _display_labels(event_labels),
    }


async def classify_article(title: str, excerpt: str | None, content_text: str | None) -> dict[str, Any]:
    body = "\n\n".join(part for part in [title, excerpt or "", content_text or ""] if part).strip()
    heuristic_topics = _heuristic_topics(body)
    heuristic_sectors = _normalize_labels(
        [SECTOR_MAP[topic] for topic in heuristic_topics if topic in SECTOR_MAP]
    )
    heuristic_event_payload = _heuristic_event_payload(body)
    heuristic_payload = {
        "topics": heuristic_topics,
        "tickers": [],
        "sectors": heuristic_sectors,
        "importance": _heuristic_importance(body),
        "sentiment": _heuristic_sentiment(body),
        "event_type": heuristic_event_payload["event_type"],
        "event_labels": heuristic_event_payload["event_labels"],
    }

    prompt = (
        "Classify the article into strict JSON with keys: "
        "{\"topics\": string[], \"tickers\": string[], \"sectors\": string[], "
        "\"importance\": \"low|medium|high\", \"sentiment\": \"negative|neutral|positive\", "
        "\"event_type\": \"earnings|dividend|capital_raise|insider_trading|management_change|regulatory|mna|analyst_view|macro_policy|other\", "
        "\"event_labels\": string[]}. "
        "Use concise normalized labels. "
        "For tickers, include only real listed stock symbols explicitly supported by the article context. "
        "Do not guess. Do not return generic uppercase words, country abbreviations, organization acronyms, product names, policy terms, or currencies such as USD. "
        "If no clear stock ticker is present, return an empty tickers array."
    )
    llm_payload = await _call_json_llm(
        NEWS_ARTICLE_CLASSIFICATION_TASK,
        "You are a strict JSON classifier for financial news.",
        f"{prompt}\n\nArticle:\n{body[:6000]}",
    )
    if not llm_payload:
        return {**heuristic_payload, "raw_payload": heuristic_payload}

    display_topics = _display_labels([str(item) for item in llm_payload.get("topics", [])])
    topics = _normalize_labels(display_topics)
    tickers = sorted({str(item).strip().upper() for item in llm_payload.get("tickers", []) if str(item).strip()})
    sectors = _normalize_labels([str(item) for item in llm_payload.get("sectors", [])])
    importance = str(llm_payload.get("importance") or heuristic_payload["importance"]).strip().lower()
    sentiment = str(llm_payload.get("sentiment") or heuristic_payload["sentiment"]).strip().lower()
    raw_payload = dict(llm_payload)
    raw_payload["display_topics"] = display_topics or heuristic_topics
    event_type = str(llm_payload.get("event_type") or heuristic_payload["event_type"]).strip().lower()
    if event_type not in EVENT_TYPE_DISPLAY_LABELS:
        event_type = heuristic_payload["event_type"]
    event_labels = _display_labels([str(item) for item in llm_payload.get("event_labels", [])])
    if not event_labels:
        event_labels = heuristic_payload["event_labels"]
    raw_payload["event_type"] = event_type
    raw_payload["event_labels"] = event_labels

    merged = {
        "topics": topics or heuristic_topics,
        "tickers": tickers,
        "sectors": sectors or heuristic_sectors,
        "importance": importance if importance in {"low", "medium", "high"} else heuristic_payload["importance"],
        "sentiment": sentiment if sentiment in {"negative", "neutral", "positive"} else heuristic_payload["sentiment"],
        "event_type": event_type,
        "event_labels": event_labels,
        "raw_payload": raw_payload,
    }
    return merged


async def summarize_article(
    title: str,
    excerpt: str | None,
    content_text: str | None,
    *,
    language: str | None = None,
) -> str | None:
    body = "\n\n".join(part for part in [title, excerpt or "", content_text or ""] if part).strip()
    if not body:
        return None

    language_instruction = (
        f"Write the summary in the article's original language. Language hint: {language.strip()}."
        if language and language.strip()
        else "Write the summary in the article's original language inferred from the article text."
    )
    prompt = (
        "Summarize this financial news article into strict JSON with one key: "
        "{\"summary\": string}. Keep it factual, concise, and suitable for a feed card. "
        f"Target {ARTICLE_SUMMARY_TARGET_MIN_WORDS}-{ARTICLE_SUMMARY_TARGET_MAX_WORDS} words, "
        "usually in 3-5 short complete sentences. "
        "Cover more of the article's key points than a typical short excerpt, including the main event, "
        "the most important context, and any notable company, market, or policy impact when present. "
        f"If the summary runs a little long, prefer complete sentences over abrupt cutoffs, but keep it under {ARTICLE_SUMMARY_SOFT_MAX_WORDS} words when possible. "
        f"{language_instruction}"
    )
    llm_payload = await _call_json_llm(
        NEWS_ARTICLE_SUMMARY_TASK,
        "You summarize financial news articles and return JSON only.",
        f"{prompt}\n\nArticle:\n{body[:6000]}",
    )
    if not llm_payload:
        return None

    summary = _normalize_summary_text(str(llm_payload.get("summary") or ""))
    if not summary:
        return None
    return summary


async def generate_quick_glance_digest(
    *,
    window_hours: int,
    article_count: int,
    highlights_target: int,
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not evidence_items:
        return None

    compact_evidence = [
        {
            "article_id": int(item["article_id"]),
            "title": _normalize_discussion_text(str(item.get("title") or ""), limit=220),
            "published_at": _normalize_discussion_text(str(item.get("published_at") or ""), limit=40) or None,
            "source_title": _normalize_discussion_text(str(item.get("source_title") or ""), limit=120) or None,
            "importance": _normalize_discussion_text(str(item.get("importance") or ""), limit=20) or None,
            "sentiment": _normalize_discussion_text(str(item.get("sentiment") or ""), limit=20) or None,
            "event_type": _normalize_discussion_text(str(item.get("event_type") or ""), limit=40) or None,
            "topics": [str(topic) for topic in item.get("topics", [])][:6],
            "tickers": [str(ticker) for ticker in item.get("tickers", [])][:6],
            "story_source_count": int(item.get("story_source_count") or 1),
            "why_relevant": [_normalize_discussion_text(str(reason), limit=160) for reason in item.get("why_relevant", [])[:3]],
            "summary_text": _normalize_discussion_text(str(item.get("summary_text") or ""), limit=900),
            "content_text": _normalize_discussion_text(str(item.get("content_text") or ""), limit=1200) or None,
        }
        for item in evidence_items
        if item.get("article_id") and str(item.get("title") or "").strip()
    ]
    if not compact_evidence:
        return None

    prompt = (
        "Return strict JSON with keys: "
        "{\"summary\": string, \"highlights\": [{\"title\": string, \"body\": string, \"article_ids\": number[]}]}.\n"
        "You are summarizing a financial news digest across multiple articles.\n"
        "Use only the provided evidence. Do not invent facts, causes, entities, dates, or article ids.\n"
        "Prioritize the most decision-relevant developments for investors: major company events, broad market or policy moves, "
        "unusual importance or sentiment clusters, and stories repeated across multiple sources.\n"
        f"Write one concise executive summary under {QUICK_GLANCE_SUMMARY_SOFT_MAX_WORDS} words.\n"
        f"Return 3-{highlights_target} highlights when the evidence supports it. Each highlight body should stay under {QUICK_GLANCE_HIGHLIGHT_SOFT_MAX_WORDS} words.\n"
        "For each highlight, article_ids must contain only article_id values from the evidence list below.\n"
        "Prefer complete, grounded statements over vague market commentary.\n"
        "Always write the summary and every highlight in Vietnamese."
    )
    llm_payload = await _call_json_llm(
        NEWS_QUICK_GLANCE_TASK,
        "You summarize multi-article financial news digests and return JSON only.",
        (
            f"{prompt}\n\n"
            f"context:\n{json.dumps({'window_hours': window_hours, 'article_count': article_count}, ensure_ascii=False)}\n\n"
            f"evidence:\n{json.dumps(compact_evidence, ensure_ascii=False)}"
        ),
    )
    if not llm_payload:
        return None

    summary = _normalize_summary_text(str(llm_payload.get("summary") or ""))
    if not summary:
        return None

    valid_article_ids = {int(item["article_id"]) for item in compact_evidence}
    highlights: list[dict[str, Any]] = []
    for raw_highlight in llm_payload.get("highlights", []):
        if not isinstance(raw_highlight, dict):
            continue
        title = _normalize_discussion_text(str(raw_highlight.get("title") or ""), limit=140)
        body = _normalize_discussion_text(str(raw_highlight.get("body") or ""), limit=420)
        article_ids: list[int] = []
        for raw_id in raw_highlight.get("article_ids", []):
            try:
                article_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if article_id in valid_article_ids and article_id not in article_ids:
                article_ids.append(article_id)
        if not title or not body or not article_ids:
            continue
        highlights.append(
            {
                "title": title,
                "body": body,
                "article_ids": article_ids,
            }
        )
        if len(highlights) >= highlights_target:
            break

    return {
        "summary": summary,
        "highlights": highlights,
    }


async def discuss_article_with_context(
    *,
    article_context: dict[str, Any],
    messages: list[dict[str, str]],
    evidence_items: list[dict[str, Any]],
    search_web: bool,
) -> dict[str, Any] | None:
    if not messages:
        return None

    compact_messages = [
        {
            "role": str(item.get("role") or "").strip().lower(),
            "content": _normalize_discussion_text(str(item.get("content") or ""), limit=1500),
        }
        for item in messages
        if str(item.get("role") or "").strip().lower() in {"user", "assistant"} and str(item.get("content") or "").strip()
    ]
    compact_evidence = [
        {
            "source_id": str(item.get("source_id") or "").strip(),
            "source_type": str(item.get("source_type") or "").strip(),
            "title": _normalize_discussion_text(str(item.get("title") or ""), limit=220),
            "domain": _normalize_discussion_text(str(item.get("domain") or ""), limit=120) or None,
            "url": _normalize_discussion_text(str(item.get("url") or ""), limit=500) or None,
            "snippet": _normalize_discussion_text(str(item.get("snippet") or ""), limit=700),
        }
        for item in evidence_items
        if str(item.get("source_id") or "").strip() and str(item.get("snippet") or item.get("title") or "").strip()
    ]
    if not compact_messages or not compact_evidence:
        return None

    prompt = (
        "Return strict JSON with keys: "
        "{\"assistant_message\": string, \"cited_source_ids\": string[], \"warning\": string|null}. "
        "You are a grounded discussion assistant for financial news. "
        "Answer only from the supplied article context and evidence. "
        "Treat earlier assistant messages in the conversation as already delivered context. "
        "For follow-up questions, answer incrementally and avoid repeating points already covered unless the user explicitly asks for a recap or the repeated point is necessary to clarify/correct the answer. "
        "Keep overlap with prior assistant answers minimal. "
        "Never invent facts, dates, quotes, or sources. "
        "If search_web is false, do not imply you searched the internet. "
        "If search_web is true and relevant web evidence exists, prefer citing at least one web source for outside context rather than defaulting back to the article alone. "
        "If multiple strong web evidence items are provided for a broad background question, cite more than one distinct web source_id instead of relying on a single outside citation. "
        "If the available evidence is insufficient, say so plainly. "
        "Avoid investment-advice phrasing and do not tell the user to buy, sell, or hold. "
        "When the answer has multiple distinct points, format them as markdown bullet lines instead of inline dash-separated prose. "
        "Every substantive factual answer must cite at least one source_id from the provided evidence. "
        "Prefer article citations for article-grounded points and web citations for outside context. "
        "Keep the answer concise and directly responsive to the latest user message."
    )
    llm_payload = await _call_json_llm(
        NEWS_ARTICLE_DISCUSSION_TASK,
        "You discuss a news article using only provided evidence and return JSON only.",
        (
            f"{prompt}\n\n"
            f"search_web: {json.dumps(bool(search_web))}\n\n"
            f"article_context:\n{json.dumps(article_context, ensure_ascii=False)}\n\n"
            f"conversation:\n{json.dumps(compact_messages, ensure_ascii=False)}\n\n"
            f"evidence:\n{json.dumps(compact_evidence, ensure_ascii=False)}"
        ),
    )
    if not llm_payload:
        return None

    assistant_message = _normalize_discussion_text(str(llm_payload.get("assistant_message") or ""), limit=4000)
    if not assistant_message:
        return None
    cited_source_ids = [
        str(item).strip()
        for item in llm_payload.get("cited_source_ids", [])
        if str(item).strip()
    ]
    warning = _normalize_discussion_text(str(llm_payload.get("warning") or ""), limit=500) or None
    return {
        "assistant_message": assistant_message,
        "cited_source_ids": cited_source_ids,
        "warning": warning,
    }


async def generate_discussion_search_queries(
    *,
    article_context: dict[str, Any],
    messages: list[dict[str, str]],
    fallback_queries: list[str],
) -> list[str] | None:
    compact_messages = [
        {
            "role": str(item.get("role") or "").strip().lower(),
            "content": _normalize_discussion_text(str(item.get("content") or ""), limit=800),
        }
        for item in messages
        if str(item.get("role") or "").strip().lower() in {"user", "assistant"} and str(item.get("content") or "").strip()
    ]
    if not compact_messages:
        return None

    prompt = (
        "Return strict JSON with one key: {\"queries\": string[]}. "
        "Generate 2-4 web-search queries to gather evidence for the user's latest question about the article topic. "
        "First infer the intent from the latest user message, such as overview, background, reason/causal, ownership, business model, financial performance, comparison, or recent updates. "
        "Use concise search-engine-friendly queries, not full sentences. "
        "Prefer entity-focused queries for overview/background requests instead of repeating the full article headline. "
        "Use article/event-anchored queries for causal or follow-up questions tied to the specific story. "
        "When useful, include both Vietnamese and English variants, entity aliases, or company-overview phrasing. "
        "Do not invent unsupported entities. Do not include search operators unless clearly useful. "
        "Queries should complement each other rather than being near-duplicates."
    )
    llm_payload = await _call_json_llm(
        NEWS_ARTICLE_DISCUSSION_QUERY_TASK,
        "You generate web-search queries for grounded article discussion and return JSON only.",
        (
            f"{prompt}\n\n"
            f"article_context:\n{json.dumps(article_context, ensure_ascii=False)}\n\n"
            f"conversation:\n{json.dumps(compact_messages, ensure_ascii=False)}\n\n"
            f"fallback_queries:\n{json.dumps(fallback_queries, ensure_ascii=False)}"
        ),
    )
    if not llm_payload:
        return None

    queries = [
        _normalize_discussion_query(str(item))
        for item in llm_payload.get("queries", [])
        if _normalize_discussion_query(str(item))
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
        if len(deduped) >= 4:
            break
    return deduped or None


async def decide_discussion_search(
    *,
    article_context: dict[str, Any],
    messages: list[dict[str, str]],
    article_content_strength: str,
) -> dict[str, Any] | None:
    compact_messages = [
        {
            "role": str(item.get("role") or "").strip().lower(),
            "content": _normalize_discussion_text(str(item.get("content") or ""), limit=800),
        }
        for item in messages
        if str(item.get("role") or "").strip().lower() in {"user", "assistant"} and str(item.get("content") or "").strip()
    ]
    if not compact_messages:
        return None

    prompt = (
        "Return strict JSON with keys: "
        "{\"intent\": string, \"subject\": string|null, \"needs_web_search\": boolean, \"reason\": string, \"confidence\": number}. "
        "Classify the latest user message for grounded article discussion. "
        "Allowed intents: recap, overview, ownership, business, financials, comparison, latest, event, generic. "
        "Set needs_web_search=true when the user is asking for broader background, ownership, company profile, comparison, latest updates, or other context that likely exceeds the article alone. "
        "Set needs_web_search=false when the question is mainly a recap or directly answerable from the article body. "
        "Infer a concise subject/entity when possible. "
        "Use the provided article_content_strength as a hint, but do not let strong article text prevent web search for clear background/ownership/profile questions."
    )
    llm_payload = await _call_json_llm(
        NEWS_ARTICLE_DISCUSSION_DECISION_TASK,
        "You decide whether grounded article discussion should trigger web search and return JSON only.",
        (
            f"{prompt}\n\n"
            f"article_content_strength: {article_content_strength}\n\n"
            f"article_context:\n{json.dumps(article_context, ensure_ascii=False)}\n\n"
            f"conversation:\n{json.dumps(compact_messages, ensure_ascii=False)}"
        ),
    )
    if not llm_payload:
        return None

    intent = _normalize_discussion_query(str(llm_payload.get("intent") or ""), limit=40).lower()
    if intent not in {"recap", "overview", "ownership", "business", "financials", "comparison", "latest", "event", "generic"}:
        intent = "generic"
    subject = _normalize_discussion_query(str(llm_payload.get("subject") or ""), limit=120) or None
    reason = _normalize_discussion_text(str(llm_payload.get("reason") or ""), limit=300)
    try:
        confidence = float(llm_payload.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    return {
        "intent": intent,
        "subject": subject,
        "needs_web_search": bool(llm_payload.get("needs_web_search")),
        "reason": reason or "",
        "confidence": confidence,
    }


async def compile_blocked_labels(blocked_topics_text: str | None) -> list[str]:
    if not blocked_topics_text or not blocked_topics_text.strip():
        return []

    raw_text = blocked_topics_text.strip()
    fallback = _normalize_labels([part for part in BLOCKED_SPLIT_PATTERN.split(raw_text) if part.strip()])

    prompt = (
        "Convert the user's blocked-topic preference text into a strict JSON payload "
        "with one key: {\"labels\": string[]}. Use concise normalized topic labels."
    )
    llm_payload = await _call_json_llm(
        NEWS_BLOCKED_LABEL_COMPILATION_TASK,
        "You extract normalized blocked-topic labels and return JSON only.",
        f"{prompt}\n\nPreference text:\n{raw_text}",
    )
    if not llm_payload:
        return fallback
    labels = _normalize_labels([str(item) for item in llm_payload.get("labels", [])])
    return labels or fallback


def matches_blocked_labels(
    blocked_labels: list[str],
    *,
    article_topics: list[str],
    article_tickers: list[str],
    title: str,
    excerpt: str | None,
    content_text: str | None,
) -> bool:
    if not blocked_labels:
        return False
    blocked = {_normalize_label(item) for item in blocked_labels if item}
    semantic_labels = {_normalize_label(item) for item in article_topics}
    semantic_labels.update(_normalize_label(item) for item in article_tickers)
    if blocked.intersection(semantic_labels):
        return True
    haystack = " ".join(part.lower() for part in [title, excerpt or "", content_text or ""] if part)
    return any(label.replace("_", " ") in haystack for label in blocked if label)
