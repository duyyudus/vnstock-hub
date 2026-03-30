from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.core.config import settings

from .discovery import extract_domain


_WHITESPACE_RE = re.compile(r"\s+")


def _compact_text(value: str | None, *, limit: int = 400) -> str:
    cleaned = _WHITESPACE_RE.sub(" ", str(value or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 1)].rstrip()}…"


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    domain: str | None


class NewsSearchProvider(Protocol):
    async def search(self, query: str, *, limit: int = 5) -> list[WebSearchResult]:
        ...


class BraveNewsSearchProvider:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str,
        base_url: str,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url

    async def search(self, query: str, *, limit: int = 5) -> list[WebSearchResult]:
        response = await self._client.get(
            self._base_url,
            params={
                "q": query,
                "count": max(1, min(limit, 10)),
                "text_decorations": 0,
                "extra_snippets": 1,
            },
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self._api_key,
            },
            timeout=settings.news_search_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        items = payload.get("web", {}).get("results") or payload.get("results") or []
        results: list[WebSearchResult] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            title = _compact_text(item.get("title"), limit=220)
            if not url or not title:
                continue
            snippet = _compact_text(
                item.get("description")
                or item.get("snippet")
                or " ".join(str(part) for part in (item.get("extra_snippets") or []) if str(part).strip()),
                limit=420,
            )
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    domain=extract_domain(url),
                )
            )
        return results


def get_news_search_provider(client: httpx.AsyncClient) -> NewsSearchProvider | None:
    provider = settings.news_search_provider.strip().lower()
    api_key = str(settings.news_search_api_key or "").strip()
    if not provider or not api_key:
        return None
    if provider == "brave":
        return BraveNewsSearchProvider(
            client,
            api_key=api_key,
            base_url=str(settings.news_search_base_url).strip() or "https://api.search.brave.com/res/v1/web/search",
        )
    return None
