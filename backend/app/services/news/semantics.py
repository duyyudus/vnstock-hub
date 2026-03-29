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
    NEWS_ARTICLE_SUMMARY_TASK,
    NEWS_BLOCKED_LABEL_COMPILATION_TASK,
)
from app.services.llm.llm_client import _build_chat_url, _extract_json_payload


logger = logging.getLogger("vnstock_hub.news")
PROVIDER_FAILURE_COOLDOWN_SECONDS = 300.0
PROVIDER_CONFIGURATION_COOLDOWN_SECONDS = 1800.0
ARTICLE_SUMMARY_TARGET_MIN_WORDS = 70
ARTICLE_SUMMARY_TARGET_MAX_WORDS = 90
ARTICLE_SUMMARY_SOFT_MAX_WORDS = 120
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
