from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import unicodedata
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import and_, delete, or_, select
import yaml

from app.core.config import settings
from app.db.database import async_session
from app.db.models import (
    BookmarkGroup,
    BookmarkStock,
    NewsArticle,
    NewsQuickGlanceDigest,
    NewsArticleSemantic,
    NewsArticleSource,
    NewsCrawlSource,
    NewsIngestionRun,
    NewsSite,
    NewsSiteFeed,
    NewsSourceSubscription,
    NewsUserPreference,
    PortfolioPosition,
)

from .discovery import (
    discover_crawl_listings,
    discover_rss_feeds,
    extract_article_payload,
    extract_domain,
    extract_links_with_selector,
    fetch_text,
    normalize_url,
    parse_feed_entries,
    validate_crawl_source,
    validate_feed,
)
from .search import get_news_search_provider
from .semantics import (
    _display_labels,
    classify_article,
    compile_blocked_labels,
    decide_discussion_search,
    discuss_article_with_context,
    generate_quick_glance_digest,
    generate_discussion_search_queries,
    matches_blocked_labels,
    summarize_article,
)


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
UTC = timezone.utc
NEWS_STALE_RUN_MINUTES = 30
CONTENT_REPAIR_MARKERS = (
    "từ khóa:",
    "đọc thêm",
    "bài viết mới nhất",
    "đọc nhiều nhất",
    "askonomy",
    "trợ lý thông tin kinh tế",
)
ALWAYS_REFRESH_DETAIL_DOMAINS = {"vneconomy.vn"}
LEGACY_FEED_TITLE_WORD_THRESHOLD = 8
LEGACY_FEED_TITLE_LENGTH_THRESHOLD = 60
NEWS_STORY_BUCKET_HOURS = 12
STORY_TITLE_STOPWORDS = {
    "and",
    "are",
    "cua",
    "cho",
    "duoc",
    "giao",
    "has",
    "issue",
    "issued",
    "khong",
    "la",
    "nam",
    "news",
    "sau",
    "says",
    "says",
    "sẽ",
    "tai",
    "the",
    "thi",
    "thong",
    "tin",
    "to",
    "trong",
    "updated",
    "ve",
    "voi",
    "với",
}
IMPORTANCE_WEIGHTS = {"high": 60, "medium": 30, "low": 10}
SOURCE_MULTIPLIER_CAP = 4
DISCUSSION_MAX_TURNS = 6
DISCUSSION_MAX_MESSAGES = DISCUSSION_MAX_TURNS * 2
DISCUSSION_MAX_MESSAGE_CHARS = 1500
DISCUSSION_MAX_ARTICLE_CHARS = 6000
DISCUSSION_MAX_SNIPPET_CHARS = 700
DISCUSSION_WEB_RESULT_LIMIT = 5
DISCUSSION_WEB_EVIDENCE_LIMIT = 3
DISCUSSION_WEB_QUERY_LIMIT = 3
DISCUSSION_WEB_UNIQUE_RESULT_LIMIT = 8
DISCUSSION_WEB_RETRY_QUERY_LIMIT = 2
QUICK_GLANCE_EVIDENCE_LIMIT = 24
QUICK_GLANCE_HIGHLIGHT_LIMIT = 7
QUICK_GLANCE_KEY_ARTICLE_LIMIT = 10
QUICK_GLANCE_REFRESH_COOLDOWN_MINUTES = 30
QUICK_GLANCE_MIN_CHANGED_STORIES = 2
DISCUSSION_BACKGROUND_INTENTS = {"overview", "ownership", "business", "financials", "comparison"}
DISCUSSION_RESULT_MIN_SCORE_FOR_BACKGROUND = 25
DISCUSSION_PROFILE_KEYWORDS = (
    "gioi thieu",
    "giới thiệu",
    "about",
    "overview",
    "company",
    "profile",
    "trang chu",
    "trang chủ",
    "homepage",
    "official",
    "consumer finance",
    "tai chinh tieu dung",
    "tài chính tiêu dùng",
)
DISCUSSION_OWNERSHIP_KEYWORDS = (
    "thuoc ai",
    "thuộc ai",
    "owner",
    "ownership",
    "shareholder",
    "parent company",
    "belongs to",
)
DISCUSSION_BUSINESS_KEYWORDS = (
    "san pham",
    "sản phẩm",
    "dich vu",
    "dịch vụ",
    "business model",
    "hoat dong",
    "hoạt động",
    "product",
    "service",
)
DISCUSSION_FINANCIAL_KEYWORDS = (
    "ket qua kinh doanh",
    "kết quả kinh doanh",
    "profit",
    "revenue",
    "financial results",
    "earnings",
    "annual report",
)
DISCUSSION_LATEST_KEYWORDS = ("moi nhat", "mới nhất", "latest", "recent", "news today", "tin moi")
DISCUSSION_EVENT_CAUSAL_KEYWORDS = ("vi sao", "vì sao", "tai sao", "tại sao", "reason", "why", "collapse", "deal")
DISCUSSION_RECAP_KEYWORDS = (
    "tom tat",
    "tóm tắt",
    "summarize",
    "summary",
    "what happened",
    "dieu gi da xay ra",
    "điều gì đã xảy ra",
    "why relevant",
    "vi sao lien quan",
    "vì sao liên quan",
)


def _compact_whitespace(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate_text(value: str | None, *, limit: int) -> str:
    cleaned = _compact_whitespace(value)
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 1)].rstrip()}…"


def _normalized_search_text(value: str | None) -> str:
    return _ascii_fold(_compact_whitespace(value)).lower()


def _significant_search_tokens(value: str | None) -> set[str]:
    lowered = _normalized_search_text(value)
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", lowered)
        if len(token) >= 3 and token not in STORY_TITLE_STOPWORDS
    }
    return tokens


def utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(tzinfo=None)


def utc_naive_to_local_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    aware = value.replace(tzinfo=UTC).astimezone(VN_TZ)
    return aware.isoformat(timespec="seconds")


def _matches_topic_filter(topic_filter: str | None, topics: list[str]) -> bool:
    if not topic_filter:
        return True
    normalized_filter = topic_filter.strip().lower()
    if not normalized_filter:
        return True
    return any(normalized_filter in str(item).strip().lower() for item in topics if item)


def _display_topics(semantics: NewsArticleSemantic | None) -> list[str]:
    if semantics is None:
        return []
    raw_payload = semantics.raw_payload if isinstance(semantics.raw_payload, dict) else {}
    for key in ("display_topics", "topics"):
        raw_topics = raw_payload.get(key)
        if isinstance(raw_topics, list):
            display_topics = _display_labels([str(item) for item in raw_topics])
            if display_topics:
                return display_topics
    return [str(item) for item in (semantics.topics or [])]


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_story_title(value: str) -> str:
    folded = _ascii_fold(value).lower()
    return re.sub(r"[^a-z0-9\s]+", " ", folded)


def _story_signature(title: str) -> str:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+", _normalize_story_title(title)):
        if len(token) < 3 or token in STORY_TITLE_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= 6:
            break
    return "-".join(tokens) if tokens else "story"


def _story_time_bucket(published_at: datetime | None) -> str:
    if published_at is None:
        return "na"
    bucket_hour = (published_at.hour // NEWS_STORY_BUCKET_HOURS) * NEWS_STORY_BUCKET_HOURS
    return published_at.replace(hour=bucket_hour, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")


def _compute_story_key(title: str, tickers: list[str], published_at: datetime | None) -> str:
    primary_ticker = (sorted({str(item).strip().upper() for item in tickers if str(item).strip()}) or ["MARKET"])[0]
    signature = _story_signature(title)
    return f"{primary_ticker}|{_story_time_bucket(published_at)}|{signature}"


def _semantic_raw_payload(semantics: NewsArticleSemantic | None) -> dict[str, Any]:
    if semantics is None or not isinstance(semantics.raw_payload, dict):
        return {}
    return semantics.raw_payload


def _semantic_event_type(semantics: NewsArticleSemantic | None) -> str | None:
    raw_payload = _semantic_raw_payload(semantics)
    value = str(raw_payload.get("event_type") or "").strip().lower()
    return value or None


def _semantic_event_labels(semantics: NewsArticleSemantic | None) -> list[str]:
    raw_payload = _semantic_raw_payload(semantics)
    values = raw_payload.get("event_labels")
    if isinstance(values, list):
        labels = _display_labels([str(item) for item in values])
        if labels:
            return labels
    event_type = _semantic_event_type(semantics)
    if not event_type:
        return []
    return [event_type.replace("_", " ")]


def _semantic_story_key(semantics: NewsArticleSemantic | None, article: NewsArticle) -> str:
    raw_payload = _semantic_raw_payload(semantics)
    story_key = str(raw_payload.get("story_key") or "").strip()
    if story_key:
        return story_key
    tickers = [str(item) for item in (semantics.tickers if semantics else [])]
    return _compute_story_key(article.title, tickers, article.published_at)


def _importance_value(importance: str | None) -> int:
    return IMPORTANCE_WEIGHTS.get(str(importance or "").strip().lower(), 0)


class NewsIngestionService:
    def __init__(self) -> None:
        self._loop_task: asyncio.Task | None = None
        self._startup_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def start_background_tasks(self) -> None:
        async with self._startup_lock:
            if self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
            await self.reconcile_stale_runs()
            await self.ensure_public_sources()
            if not settings.news_ingestion_enabled:
                return
            if self._loop_task and not self._loop_task.done():
                return
            self._loop_task = asyncio.create_task(self._run_loop())

    async def stop_background_tasks(self) -> None:
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        self._loop_task = None
        await self._client.aclose()

    async def ensure_public_sources(self) -> None:
        config_path = Path(settings.news_sources_yaml_path)
        if not config_path.exists():
            return

        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        public_sources = payload.get("public_sources") or []
        configured_public_feed_urls: set[str] = set()
        async with async_session() as session:
            for site_payload in public_sources:
                homepage_url = normalize_url(str(site_payload.get("homepage_url") or "").strip())
                if not homepage_url:
                    continue
                display_name = str(site_payload.get("display_name") or "").strip() or None
                site = await self._get_or_create_site(
                    session,
                    homepage_url=homepage_url,
                    display_name=display_name,
                    is_public=True,
                )
                for feed_payload in site_payload.get("feeds") or []:
                    feed_url = normalize_url(str(feed_payload.get("feed_url") or "").strip())
                    if not feed_url:
                        continue
                    configured_public_feed_urls.add(feed_url)
                    await self._get_or_create_feed(
                        session,
                        site_id=site.id,
                        feed_url=feed_url,
                        title=str(feed_payload.get("title") or "").strip() or display_name,
                        kind=str(feed_payload.get("kind") or "rss"),
                        discovery_method="seed",
                        validation_status="valid",
                        is_public=True,
                        poll_interval_minutes=int(
                            feed_payload.get("poll_interval_minutes") or self.get_default_poll_interval_minutes()
                        ),
                    )

            # Seeded public feeds should mirror the YAML config. If a feed was removed
            # from the config, remove the stale seeded record from the DB as well.
            stale_public_seed_feeds = (
                await session.execute(
                    select(NewsSiteFeed).where(
                        NewsSiteFeed.is_public.is_(True),
                        NewsSiteFeed.discovery_method == "seed",
                    )
                )
            ).scalars().all()
            for feed in stale_public_seed_feeds:
                if feed.feed_url not in configured_public_feed_urls:
                    await session.delete(feed)

            remaining_feeds = (await session.execute(select(NewsSiteFeed))).scalars().all()
            remaining_crawl_sources = (await session.execute(select(NewsCrawlSource))).scalars().all()
            remaining_feed_site_ids = {int(feed.site_id) for feed in remaining_feeds}
            remaining_crawl_site_ids = {int(source.site_id) for source in remaining_crawl_sources}

            site_rows = (await session.execute(select(NewsSite))).scalars().all()
            for site in site_rows:
                has_sources = site.id in remaining_feed_site_ids or site.id in remaining_crawl_site_ids
                if not has_sources:
                    await session.delete(site)
                    continue

                site.is_public = any(feed.is_public for feed in remaining_feeds if int(feed.site_id) == int(site.id)) or any(
                    source.is_public for source in remaining_crawl_sources if int(source.site_id) == int(site.id)
                )
            await session.commit()

    async def normalize_legacy_rss_titles(self) -> int:
        async with async_session() as session:
            rows = (
                await session.execute(
                    select(NewsSiteFeed, NewsSite)
                    .join(NewsSite, NewsSite.id == NewsSiteFeed.site_id)
                    .order_by(NewsSiteFeed.id.asc())
                )
            ).all()
            changed = False
            changed_count = 0
            for feed, site in rows:
                if not self._should_normalize_legacy_feed_title(feed.title, site):
                    continue
                feed.title = site.domain
                changed = True
                changed_count += 1
            if changed:
                await session.commit()
            return changed_count

    def get_default_poll_interval_minutes(self) -> int:
        payload = self._load_app_settings_yaml()
        news_settings = payload.get("news")
        if isinstance(news_settings, dict):
            value = news_settings.get("default_poll_interval_minutes")
            try:
                if value is not None:
                    return max(5, int(value))
            except (TypeError, ValueError):
                pass
        return int(settings.news_default_poll_interval_minutes)

    def get_admin_config(self) -> dict[str, Any]:
        return {
            "default_poll_interval_minutes": self.get_default_poll_interval_minutes(),
        }

    def update_admin_config(self, *, default_poll_interval_minutes: int | None = None) -> dict[str, Any]:
        payload = self._load_app_settings_yaml()
        news_settings = payload.get("news")
        if not isinstance(news_settings, dict):
            news_settings = {}
        if default_poll_interval_minutes is not None:
            news_settings["default_poll_interval_minutes"] = max(5, int(default_poll_interval_minutes))
        payload["news"] = news_settings
        self._write_app_settings_yaml(payload)
        return self.get_admin_config()

    async def apply_default_poll_interval_to_existing_sources(self) -> int:
        default_interval = self.get_default_poll_interval_minutes()
        now = utc_now()
        changed_count = 0

        async with async_session() as session:
            feeds = (await session.execute(select(NewsSiteFeed))).scalars().all()
            for feed in feeds:
                if int(feed.poll_interval_minutes or 0) == default_interval:
                    continue
                feed.poll_interval_minutes = default_interval
                feed.next_poll_at = now
                changed_count += 1

            crawl_sources = (await session.execute(select(NewsCrawlSource))).scalars().all()
            for source in crawl_sources:
                if int(source.poll_interval_minutes or 0) == default_interval:
                    continue
                source.poll_interval_minutes = default_interval
                source.next_poll_at = now
                changed_count += 1

            if changed_count:
                await session.commit()

        return changed_count

    async def set_all_admin_sources_enabled(self, enabled: bool) -> int:
        changed_count = 0

        async with async_session() as session:
            feeds = (await session.execute(select(NewsSiteFeed))).scalars().all()
            for feed in feeds:
                if bool(feed.enabled) == enabled:
                    continue
                feed.enabled = enabled
                changed_count += 1

            crawl_sources = (await session.execute(select(NewsCrawlSource))).scalars().all()
            for source in crawl_sources:
                if bool(source.enabled) == enabled:
                    continue
                source.enabled = enabled
                changed_count += 1

            if changed_count:
                await session.commit()

        return changed_count

    async def set_admin_source_enabled(self, *, source_type: str, source_id: int, enabled: bool) -> dict[str, Any]:
        async with async_session() as session:
            if source_type == "rss":
                feed = await session.get(NewsSiteFeed, source_id)
                if feed is None:
                    raise ValueError("News source not found")
                feed.enabled = enabled
                await session.commit()
                return await self._serialize_feed_source_admin(session, feed)

            if source_type == "crawl":
                source = await session.get(NewsCrawlSource, source_id)
                if source is None:
                    raise ValueError("News source not found")
                source.enabled = enabled
                await session.commit()
                return await self._serialize_crawl_source_admin(session, source)

            raise ValueError("Unsupported source type")

    async def discover_rss_feeds(self, homepage_url: str) -> list[dict[str, Any]]:
        return await discover_rss_feeds(self._client, homepage_url)

    async def discover_crawl_listings(self, homepage_url: str) -> list[dict[str, Any]]:
        return await discover_crawl_listings(self._client, homepage_url)

    async def validate_rss_feed(self, feed_url: str, site_url: str | None = None) -> dict[str, Any]:
        return await validate_feed(self._client, feed_url, site_url=site_url)

    async def validate_crawl_source(
        self,
        *,
        listing_url: str,
        article_link_selector: str,
        content_selector: str,
        excerpt_selector: str | None = None,
    ) -> dict[str, Any]:
        return await validate_crawl_source(
            self._client,
            listing_url=listing_url,
            article_link_selector=article_link_selector,
            content_selector=content_selector,
            excerpt_selector=excerpt_selector,
        )

    async def create_rss_source(
        self,
        *,
        user_id: int,
        feed_url: str,
        site_url: str | None = None,
        poll_interval_minutes: int | None = None,
    ) -> dict[str, Any]:
        validated = await self.validate_rss_feed(feed_url, site_url=site_url)
        homepage_url = normalize_url(site_url or validated.get("site_url") or feed_url)
        async with async_session() as session:
            site = await self._get_or_create_site(
                session,
                homepage_url=homepage_url,
                display_name=extract_domain(homepage_url),
                is_public=False,
            )
            feed = await self._get_or_create_feed(
                session,
                site_id=site.id,
                feed_url=validated["feed_url"],
                title=validated.get("feed_title") or extract_domain(homepage_url),
                kind=validated.get("kind") or "rss",
                discovery_method="manual",
                validation_status="valid",
                is_public=False,
                poll_interval_minutes=poll_interval_minutes or self.get_default_poll_interval_minutes(),
            )
            await self._upsert_subscription(session, user_id=user_id, site_feed_id=feed.id, crawl_source_id=None)
            await session.commit()
            return await self._serialize_feed_source(session, feed.id, user_id=user_id)

    async def create_crawl_source(
        self,
        *,
        user_id: int,
        listing_url: str,
        article_link_selector: str,
        content_selector: str,
        excerpt_selector: str | None = None,
        pagination_config: dict[str, Any] | None = None,
        poll_interval_minutes: int | None = None,
    ) -> dict[str, Any]:
        await self.validate_crawl_source(
            listing_url=listing_url,
            article_link_selector=article_link_selector,
            content_selector=content_selector,
            excerpt_selector=excerpt_selector,
        )
        normalized_listing_url = normalize_url(listing_url)
        homepage_url = f"{httpx.URL(normalized_listing_url).scheme}://{httpx.URL(normalized_listing_url).host}"
        async with async_session() as session:
            site = await self._get_or_create_site(
                session,
                homepage_url=homepage_url,
                display_name=extract_domain(homepage_url),
                is_public=False,
            )
            source = await self._get_or_create_crawl_source(
                session,
                site_id=site.id,
                listing_url=normalized_listing_url,
                article_link_selector=article_link_selector,
                content_selector=content_selector,
                excerpt_selector=excerpt_selector,
                pagination_config=pagination_config,
                validation_status="valid",
                is_public=False,
                poll_interval_minutes=poll_interval_minutes or self.get_default_poll_interval_minutes(),
            )
            await self._upsert_subscription(session, user_id=user_id, site_feed_id=None, crawl_source_id=source.id)
            await session.commit()
            return await self._serialize_crawl_source(session, source.id, user_id=user_id)

    async def list_user_sources(self, user_id: int) -> list[dict[str, Any]]:
        async with async_session() as session:
            feed_rows = (
                await session.execute(
                    select(NewsSourceSubscription)
                    .where(
                        NewsSourceSubscription.user_id == user_id,
                        NewsSourceSubscription.site_feed_id.is_not(None),
                    )
                    .order_by(NewsSourceSubscription.id.asc())
                )
            ).scalars().all()
            crawl_rows = (
                await session.execute(
                    select(NewsSourceSubscription)
                    .where(
                        NewsSourceSubscription.user_id == user_id,
                        NewsSourceSubscription.crawl_source_id.is_not(None),
                    )
                    .order_by(NewsSourceSubscription.id.asc())
                )
            ).scalars().all()
            items: list[dict[str, Any]] = []
            for row in feed_rows:
                items.append(await self._serialize_feed_source(session, int(row.site_feed_id), user_id=user_id))
            for row in crawl_rows:
                items.append(await self._serialize_crawl_source(session, int(row.crawl_source_id), user_id=user_id))
            return items

    async def get_admin_status(self, *, user_id: int | None = None) -> dict[str, Any]:
        await self.reconcile_stale_runs()
        async with async_session() as session:
            site_rows = (await session.execute(select(NewsSite).order_by(NewsSite.id.asc()))).scalars().all()
            feed_rows = (await session.execute(select(NewsSiteFeed).order_by(NewsSiteFeed.id.asc()))).scalars().all()
            crawl_rows = (await session.execute(select(NewsCrawlSource).order_by(NewsCrawlSource.id.asc()))).scalars().all()
            article_rows = (await session.execute(select(NewsArticle))).scalars().all()
            run_rows = (
                await session.execute(
                    select(NewsIngestionRun).order_by(NewsIngestionRun.started_at.desc(), NewsIngestionRun.id.desc()).limit(20)
                )
            ).scalars().all()
            successful_runs = sum(1 for row in run_rows if row.status == "succeeded")
            failed_runs = sum(1 for row in run_rows if row.status == "failed")
            active_run_rows = (
                await session.execute(
                    select(NewsIngestionRun).where(
                        NewsIngestionRun.status == "running",
                        NewsIngestionRun.finished_at.is_(None),
                    )
                )
            ).scalars().all()

            feed_items = [await self._serialize_feed_source_admin(session, row, user_id=user_id) for row in feed_rows]
            crawl_items = [await self._serialize_crawl_source_admin(session, row, user_id=user_id) for row in crawl_rows]
            runs = [await self._serialize_run_admin(session, row) for row in run_rows]

            return {
                "worker_running": bool(self._loop_task and not self._loop_task.done()),
                "site_count": len(site_rows),
                "rss_source_count": len(feed_items),
                "crawl_source_count": len(crawl_items),
                "article_count": len(article_rows),
                "run_count": len(runs),
                "active_run_count": len(active_run_rows),
                "successful_run_count": successful_runs,
                "failed_run_count": failed_runs,
                "rss_sources": feed_items,
                "crawl_sources": crawl_items,
                "recent_runs": runs,
            }

    async def reconcile_stale_runs(self) -> int:
        stale_before = utc_now() - timedelta(minutes=NEWS_STALE_RUN_MINUTES)
        async with async_session() as session:
            stale_runs = (
                await session.execute(
                    select(NewsIngestionRun).where(
                        NewsIngestionRun.status == "running",
                        NewsIngestionRun.finished_at.is_(None),
                        NewsIngestionRun.started_at < stale_before,
                    )
                )
            ).scalars().all()
            if not stale_runs:
                return 0

            finished_at = utc_now()
            for run in stale_runs:
                run.status = "failed"
                if not run.error:
                    run.error = "Marked failed after stale running state exceeded monitoring threshold."
                run.finished_at = finished_at

            await session.commit()
            return len(stale_runs)

    async def trigger_admin_run(
        self,
        *,
        source_type: str | None = None,
        source_id: int | None = None,
        public_only: bool = False,
    ) -> dict[str, Any]:
        if source_type and source_id is None:
            raise ValueError("source_id is required when source_type is provided")

        if source_type == "rss" and source_id is not None:
            async with async_session() as session:
                feed = await session.get(NewsSiteFeed, source_id)
                if feed is None:
                    raise ValueError("News source not found")
                if not bool(feed.enabled):
                    return {"triggered": 0, "message": "RSS source is disabled."}
            await self._ingest_feed(source_id)
            return {"triggered": 1, "message": "RSS source ingested."}
        if source_type == "crawl" and source_id is not None:
            async with async_session() as session:
                source = await session.get(NewsCrawlSource, source_id)
                if source is None:
                    raise ValueError("News source not found")
                if not bool(source.enabled):
                    return {"triggered": 0, "message": "Crawl source is disabled."}
            await self._ingest_crawl_source(source_id)
            return {"triggered": 1, "message": "Crawl source ingested."}

        async with async_session() as session:
            feed_stmt = select(NewsSiteFeed).where(
                NewsSiteFeed.validation_status == "valid",
                NewsSiteFeed.enabled.is_(True),
            )
            crawl_stmt = select(NewsCrawlSource).where(
                NewsCrawlSource.validation_status == "valid",
                NewsCrawlSource.enabled.is_(True),
            )
            if public_only:
                feed_stmt = feed_stmt.where(NewsSiteFeed.is_public.is_(True))
                crawl_stmt = crawl_stmt.where(NewsCrawlSource.is_public.is_(True))
            feed_ids = [row.id for row in (await session.execute(feed_stmt.order_by(NewsSiteFeed.id.asc()))).scalars().all()]
            crawl_ids = [row.id for row in (await session.execute(crawl_stmt.order_by(NewsCrawlSource.id.asc()))).scalars().all()]

        for feed_id in feed_ids:
            await self._ingest_feed(feed_id)
        for crawl_id in crawl_ids:
            await self._ingest_crawl_source(crawl_id)

        triggered = len(feed_ids) + len(crawl_ids)
        scope = "public" if public_only else "all"
        return {"triggered": triggered, "message": f"Triggered ingestion for {triggered} {scope} sources."}

    async def update_source(
        self,
        *,
        user_id: int,
        source_type: str,
        source_id: int,
        enabled: bool | None = None,
        title: str | None = None,
        poll_interval_minutes: int | None = None,
    ) -> dict[str, Any]:
        async with async_session() as session:
            if source_type == "rss":
                subscription = await self._get_subscription(session, user_id=user_id, site_feed_id=source_id)
                if enabled is not None:
                    subscription.enabled = enabled
                if title is not None:
                    feed = await session.get(NewsSiteFeed, source_id)
                    if feed is None:
                        raise ValueError("News source not found")
                    feed.title = title.strip() or feed.title
                if poll_interval_minutes is not None:
                    feed = await session.get(NewsSiteFeed, source_id)
                    if feed is None:
                        raise ValueError("News source not found")
                    feed.poll_interval_minutes = poll_interval_minutes
                    feed.next_poll_at = utc_now()
                await session.commit()
                return await self._serialize_feed_source(session, source_id, user_id=user_id)

            if source_type == "crawl":
                subscription = await self._get_subscription(session, user_id=user_id, crawl_source_id=source_id)
                if enabled is not None:
                    subscription.enabled = enabled
                if title is not None:
                    source = await session.get(NewsCrawlSource, source_id)
                    if source is None:
                        raise ValueError("News source not found")
                    site = await session.get(NewsSite, source.site_id)
                    if site is not None and title.strip():
                        site.display_name = title.strip()
                if poll_interval_minutes is not None:
                    source = await session.get(NewsCrawlSource, source_id)
                    if source is None:
                        raise ValueError("News source not found")
                    source.poll_interval_minutes = poll_interval_minutes
                    source.next_poll_at = utc_now()
                await session.commit()
                return await self._serialize_crawl_source(session, source_id, user_id=user_id)

            raise ValueError("Unsupported source type")

    async def delete_source(self, *, user_id: int, source_type: str, source_id: int) -> None:
        async with async_session() as session:
            if source_type == "rss":
                subscription = await self._get_subscription(session, user_id=user_id, site_feed_id=source_id)
                feed = await session.get(NewsSiteFeed, source_id)
                if feed is None:
                    raise ValueError("News source not found")
                site_id = int(feed.site_id)
                is_public = bool(feed.is_public)
            elif source_type == "crawl":
                subscription = await self._get_subscription(session, user_id=user_id, crawl_source_id=source_id)
                source = await session.get(NewsCrawlSource, source_id)
                if source is None:
                    raise ValueError("News source not found")
                site_id = int(source.site_id)
                is_public = bool(source.is_public)
            else:
                raise ValueError("Unsupported source type")
            await session.delete(subscription)
            await session.flush()

            if not is_public:
                await self._maybe_delete_orphaned_private_source(session, source_type=source_type, source_id=source_id)
                await self._cleanup_orphaned_site(session, site_id=site_id)

            await session.commit()

    async def delete_source_admin(self, *, source_type: str, source_id: int) -> None:
        async with async_session() as session:
            await self._hard_delete_source(session, source_type=source_type, source_id=source_id)
            await session.commit()

    async def get_user_preferences(self, user_id: int) -> dict[str, Any]:
        async with async_session() as session:
            preference = await self._get_or_create_preference(session, user_id)
            await session.commit()
            return self._serialize_preference(preference)

    async def update_user_preferences(self, user_id: int, blocked_topics_text: str | None) -> dict[str, Any]:
        blocked_labels = await compile_blocked_labels(blocked_topics_text)
        async with async_session() as session:
            preference = await self._get_or_create_preference(session, user_id)
            preference.blocked_topics_text = blocked_topics_text.strip() if blocked_topics_text else None
            preference.blocked_labels = blocked_labels
            await session.commit()
            await session.refresh(preference)
            return self._serialize_preference(preference)

    async def _load_user_interest_context(
        self,
        session,
        *,
        user_id: int | None,
        bookmark_group_id: int | None = None,
    ) -> dict[str, set[str]]:
        if user_id is None:
            return {
                "portfolio_tickers": set(),
                "bookmark_tickers": set(),
            }

        portfolio_tickers = {
            str(row[0]).strip().upper()
            for row in (
                await session.execute(
                    select(PortfolioPosition.ticker).where(PortfolioPosition.user_id == user_id)
                )
            ).all()
            if str(row[0]).strip()
        }

        bookmark_stmt = (
            select(BookmarkStock.ticker)
            .join(BookmarkGroup, BookmarkGroup.id == BookmarkStock.group_id)
            .where(BookmarkGroup.user_id == user_id)
        )
        if bookmark_group_id is not None:
            bookmark_group = await session.get(BookmarkGroup, bookmark_group_id)
            if bookmark_group is None or int(bookmark_group.user_id) != int(user_id):
                raise ValueError("Bookmark group not found")
            bookmark_stmt = bookmark_stmt.where(BookmarkGroup.id == bookmark_group_id)
        bookmark_tickers = {
            str(row[0]).strip().upper()
            for row in (await session.execute(bookmark_stmt)).all()
            if str(row[0]).strip()
        }

        return {
            "portfolio_tickers": portfolio_tickers,
            "bookmark_tickers": bookmark_tickers,
        }

    def _parse_offset_cursor(self, cursor: str | None) -> int:
        if not cursor:
            return 0
        if cursor.startswith("offset:"):
            try:
                return max(0, int(cursor.split(":", 1)[1]))
            except (TypeError, ValueError):
                return 0
        return 0

    def _build_why_relevant(
        self,
        *,
        matched_portfolio_tickers: list[str],
        matched_bookmark_tickers: list[str],
        event_type: str | None,
        source_count: int,
    ) -> list[str]:
        reasons: list[str] = []
        if matched_portfolio_tickers:
            joined = ", ".join(matched_portfolio_tickers[:3])
            reasons.append(f"Matches your portfolio: {joined}")
        if matched_bookmark_tickers:
            joined = ", ".join(matched_bookmark_tickers[:3])
            reasons.append(f"Matches your bookmarks: {joined}")
        if source_count > 1:
            reasons.append(f"Covered by {source_count} sources")
        return reasons

    def _compute_relevance_score(
        self,
        *,
        matched_portfolio_tickers: list[str],
        matched_bookmark_tickers: list[str],
        importance: str | None,
        published_at: str | None,
        source_count: int,
    ) -> int:
        score = 0
        score += 130 * len(matched_portfolio_tickers[:3])
        score += 90 * len(matched_bookmark_tickers[:3])
        score += _importance_value(importance)
        score += min(source_count, SOURCE_MULTIPLIER_CAP) * 8

        if published_at:
            try:
                published_dt = datetime.fromisoformat(published_at)
            except ValueError:
                published_dt = None
            if published_dt is not None:
                age_hours = max(0.0, (utc_now() - published_dt).total_seconds() / 3600.0)
                if age_hours <= 6:
                    score += 40
                elif age_hours <= 24:
                    score += 25
                elif age_hours <= 72:
                    score += 12
        return score

    def _build_feed_payload(
        self,
        article: NewsArticle,
        *,
        sources: list[dict[str, Any]],
        semantics: NewsArticleSemantic | None,
        interest_context: dict[str, set[str]] | None = None,
        include_content: bool = False,
    ) -> dict[str, Any]:
        interest_context = interest_context or {
            "portfolio_tickers": set(),
            "bookmark_tickers": set(),
        }
        semantic_tickers = sorted({str(item).strip().upper() for item in (semantics.tickers if semantics else []) if str(item).strip()})
        matched_portfolio_tickers = sorted(set(semantic_tickers) & interest_context["portfolio_tickers"])
        matched_bookmark_tickers = sorted(set(semantic_tickers) & interest_context["bookmark_tickers"])
        matched_tickers = sorted(set(matched_portfolio_tickers + matched_bookmark_tickers))
        source_count = len(
            {
                f"{item.get('source_type')}:{item.get('source_id')}"
                for item in sources
                if item and item.get("source_id") is not None
            }
        ) or len({str(item.get("article_url") or "") for item in sources if item})
        event_type = _semantic_event_type(semantics)
        payload = self._serialize_feed_item(
            article,
            sources=sources,
            semantics=semantics,
            include_content=include_content,
        )
        payload["matched_tickers"] = matched_tickers
        payload["why_relevant"] = self._build_why_relevant(
            matched_portfolio_tickers=matched_portfolio_tickers,
            matched_bookmark_tickers=matched_bookmark_tickers,
            event_type=event_type,
            source_count=source_count,
        )
        payload["_matched_portfolio_tickers"] = matched_portfolio_tickers
        payload["_matched_bookmark_tickers"] = matched_bookmark_tickers
        payload["_sort_score"] = self._compute_relevance_score(
            matched_portfolio_tickers=matched_portfolio_tickers,
            matched_bookmark_tickers=matched_bookmark_tickers,
            importance=payload.get("importance"),
            published_at=payload.get("published_at"),
            source_count=source_count,
        )
        return payload

    def _annotate_story_groups(self, items: list[dict[str, Any]], *, group_by: str) -> list[dict[str, Any]]:
        if not items:
            return []

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        order: list[str] = []
        for item in items:
            key = str(item.get("story_key") or f"article:{item['id']}")
            if key not in grouped:
                order.append(key)
            grouped[key].append(item)

        annotated_items: list[dict[str, Any]] = []
        for key in order:
            cluster = grouped[key]
            unique_sources = {
                f"{source.get('source_type')}:{source.get('source_id')}"
                for cluster_item in cluster
                for source in (cluster_item.get("sources") or [])
                if source and source.get("source_id") is not None
            }
            related_ids = [int(cluster_item["id"]) for cluster_item in cluster[1:]]
            lead = dict(cluster[0])
            lead["story_source_count"] = len(unique_sources) or 1
            lead["related_article_ids"] = related_ids
            if len(cluster) > 1 and not any(reason.startswith("Clustered from ") for reason in lead.get("why_relevant", [])):
                lead["why_relevant"] = [*lead.get("why_relevant", []), f"Clustered from {len(cluster)} related articles"]
            if group_by == "story":
                annotated_items.append(lead)
                continue

            for cluster_item in cluster:
                item_payload = dict(cluster_item)
                item_payload["story_source_count"] = lead["story_source_count"]
                item_payload["related_article_ids"] = [
                    int(candidate["id"])
                    for candidate in cluster
                    if int(candidate["id"]) != int(cluster_item["id"])
                ]
                annotated_items.append(item_payload)
        return annotated_items

    def _story_sort_key(self, item: dict[str, Any]) -> tuple[int, str, int]:
        return (
            int(item.get("_sort_score") or 0),
            str(item.get("published_at") or ""),
            int(item.get("id") or 0),
        )

    def _quick_glance_viewer_key(self, user_id: int | None) -> str:
        return f"user:{user_id}" if user_id is not None else "public"

    def _feed_item_source_title(self, item: dict[str, Any]) -> str | None:
        source_title = _compact_whitespace(item.get("source_title"))
        if source_title:
            return source_title
        source_labels = item.get("source_labels") or []
        for label in source_labels:
            normalized = _compact_whitespace(label)
            if normalized:
                return normalized
        for source in item.get("sources") or []:
            if not isinstance(source, dict):
                continue
            normalized = _compact_whitespace(source.get("label"))
            if normalized:
                return normalized
        return None

    def _build_quick_glance_evidence_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "article_id": int(item["id"]),
            "title": item.get("title"),
            "published_at": item.get("published_at"),
            "source_title": self._feed_item_source_title(item),
            "importance": item.get("importance"),
            "sentiment": item.get("sentiment"),
            "event_type": item.get("event_type"),
            "topics": [str(topic) for topic in item.get("topics", [])],
            "tickers": [str(ticker) for ticker in item.get("tickers", [])],
            "story_source_count": int(item.get("story_source_count") or 1),
            "why_relevant": [str(reason) for reason in item.get("why_relevant", [])],
            "summary_text": item.get("llm_summary") or item.get("original_excerpt") or item.get("excerpt") or "",
            "content_text": _truncate_text(item.get("content_text"), limit=1200),
        }

    def _build_quick_glance_fingerprint(
        self,
        items: list[dict[str, Any]],
        evidence_items: list[dict[str, Any]],
    ) -> str:
        signature_payload = {
            "article_ids": [
                {
                    "id": int(item["id"]),
                    "article_updated_at": item.get("_article_updated_at"),
                    "semantic_updated_at": item.get("_semantic_updated_at"),
                    "story_key": item.get("story_key"),
                }
                for item in items
            ],
            "evidence_ids": [
                {
                    "article_id": int(item["article_id"]),
                    "summary_text": item.get("summary_text"),
                    "content_text": item.get("content_text"),
                }
                for item in evidence_items
            ],
        }
        return hashlib.sha256(repr(signature_payload).encode("utf-8")).hexdigest()

    def _parse_iso_timestamp_to_utc_naive(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value))
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(UTC).replace(tzinfo=None)

    def _quick_glance_story_activity_at(self, item: dict[str, Any]) -> datetime | None:
        timestamps = [
            self._parse_iso_timestamp_to_utc_naive(item.get("published_at")),
            self._parse_iso_timestamp_to_utc_naive(item.get("_article_updated_at")),
            self._parse_iso_timestamp_to_utc_naive(item.get("_semantic_updated_at")),
        ]
        candidates = [timestamp for timestamp in timestamps if timestamp is not None]
        return max(candidates) if candidates else None

    def _quick_glance_story_delta(
        self,
        items: list[dict[str, Any]],
        *,
        since: datetime,
    ) -> tuple[int, bool]:
        changed_story_keys: set[str] = set()
        high_importance_story_keys: set[str] = set()

        for item in items:
            story_activity_at = self._quick_glance_story_activity_at(item)
            if story_activity_at is None or story_activity_at <= since:
                continue
            story_key = str(item.get("story_key") or f"article:{int(item.get('id') or 0)}")
            changed_story_keys.add(story_key)
            if str(item.get("importance") or "").strip().lower() == "high":
                high_importance_story_keys.add(story_key)

        return len(changed_story_keys), bool(changed_story_keys & high_importance_story_keys)

    def _should_regenerate_quick_glance_digest(
        self,
        *,
        now: datetime,
        cached_digest: NewsQuickGlanceDigest,
        items: list[dict[str, Any]],
    ) -> bool:
        changed_story_count, has_high_importance_story = self._quick_glance_story_delta(
            items,
            since=cached_digest.generated_at,
        )
        if has_high_importance_story:
            return True
        cooldown_elapsed = now - cached_digest.generated_at >= timedelta(minutes=QUICK_GLANCE_REFRESH_COOLDOWN_MINUTES)
        return cooldown_elapsed and changed_story_count >= QUICK_GLANCE_MIN_CHANGED_STORIES

    def _serialize_quick_glance_key_article(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(item["id"]),
            "title": str(item.get("title") or ""),
            "published_at": item.get("published_at"),
            "canonical_url": str(item.get("canonical_url") or ""),
            "source_title": self._feed_item_source_title(item),
            "importance": item.get("importance"),
            "event_type": item.get("event_type"),
            "story_source_count": int(item.get("story_source_count") or 1),
        }

    def _is_high_importance_story(self, item: dict[str, Any]) -> bool:
        return str(item.get("importance") or "").strip().lower() == "high"

    def _select_quick_glance_evidence_story_items(
        self,
        story_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        high_priority_items = [item for item in story_items if self._is_high_importance_story(item)]
        selected: list[dict[str, Any]] = list(high_priority_items)
        seen_ids = {int(item["id"]) for item in selected if item.get("id") is not None}

        for item in story_items:
            article_id = item.get("id")
            if article_id is None or int(article_id) in seen_ids:
                continue
            if len(selected) >= QUICK_GLANCE_EVIDENCE_LIMIT:
                break
            seen_ids.add(int(article_id))
            selected.append(item)
        return selected

    def _select_quick_glance_key_articles(
        self,
        story_items: list[dict[str, Any]],
        *,
        highlighted_article_ids: list[int],
    ) -> list[dict[str, Any]]:
        story_map = {
            int(item["id"]): item
            for item in story_items
            if item.get("id") is not None
        }
        selected: list[dict[str, Any]] = []
        seen: set[int] = set()
        for article_id in highlighted_article_ids:
            item = story_map.get(int(article_id))
            if item is None or int(item["id"]) in seen:
                continue
            seen.add(int(item["id"]))
            selected.append(self._serialize_quick_glance_key_article(item))
        for item in story_items:
            article_id = int(item["id"])
            if article_id in seen or not self._is_high_importance_story(item):
                continue
            seen.add(article_id)
            selected.append(self._serialize_quick_glance_key_article(item))
        for item in story_items:
            article_id = int(item["id"])
            if article_id in seen:
                continue
            if len(selected) >= QUICK_GLANCE_KEY_ARTICLE_LIMIT:
                break
            seen.add(article_id)
            selected.append(self._serialize_quick_glance_key_article(item))
        return selected

    def _hydrate_quick_glance_payload(
        self,
        cache_row: NewsQuickGlanceDigest,
        *,
        cache_hit: bool,
    ) -> dict[str, Any]:
        payload = dict(cache_row.payload or {})
        return {
            "window_hours": int(payload.get("window_hours") or cache_row.window_hours),
            "article_count": int(payload.get("article_count") or 0),
            "oldest_article_at": payload.get("oldest_article_at"),
            "newest_article_at": payload.get("newest_article_at"),
            "generated_at": utc_naive_to_local_iso(cache_row.generated_at),
            "cache_hit": cache_hit,
            "summary": payload.get("summary"),
            "highlights": [
                {
                    "title": str(item.get("title") or ""),
                    "body": str(item.get("body") or ""),
                    "article_ids": [int(article_id) for article_id in item.get("article_ids", []) if article_id is not None],
                }
                for item in payload.get("highlights", [])
                if isinstance(item, dict)
            ],
            "key_articles": [
                {
                    "id": int(item["id"]),
                    "title": str(item.get("title") or ""),
                    "published_at": item.get("published_at"),
                    "canonical_url": str(item.get("canonical_url") or ""),
                    "source_title": item.get("source_title"),
                    "importance": item.get("importance"),
                    "event_type": item.get("event_type"),
                    "story_source_count": int(item.get("story_source_count") or 1),
                }
                for item in payload.get("key_articles", [])
                if isinstance(item, dict) and item.get("id") is not None
            ],
        }

    async def _collect_feed_items(
        self,
        session,
        *,
        user_id: int | None,
        source: str | None = None,
        ticker: str | None = None,
        topic: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        scope: str = "all",
        bookmark_group_id: int | None = None,
        event_type: str | None = None,
        importance: str | None = None,
        include_content: bool = False,
    ) -> list[dict[str, Any]]:
        source_filter = source.strip().lower() if source else None
        ticker_filter = ticker.strip().upper() if ticker else None
        topic_filter = topic.strip().lower() if topic else None
        event_type_filter = event_type.strip().lower() if event_type else None
        importance_filter = importance.strip().lower() if importance else None
        scope_mode = scope if user_id is not None and scope in {"all", "portfolio", "bookmarks"} else "all"

        blocked_labels = await self._load_blocked_labels(session, user_id)
        interest_context = await self._load_user_interest_context(
            session,
            user_id=user_id,
            bookmark_group_id=bookmark_group_id,
        )
        stmt = select(NewsArticle).order_by(NewsArticle.published_at.desc().nullslast(), NewsArticle.id.desc())
        if date_from:
            stmt = stmt.where(NewsArticle.published_at.is_not(None), NewsArticle.published_at >= date_from)
        if date_to:
            stmt = stmt.where(NewsArticle.published_at.is_not(None), NewsArticle.published_at <= date_to)
        articles = (await session.execute(stmt)).scalars().all()
        semantic_map = await self._load_semantics_map(session, [article.id for article in articles])
        source_map = await self._load_article_sources_map(session, [article.id for article in articles], user_id=user_id)

        items: list[dict[str, Any]] = []
        for article in articles:
            sources = source_map.get(article.id, [])
            if not sources:
                continue
            semantics = semantic_map.get(article.id)
            normalized_topics = [str(item) for item in (semantics.topics if semantics else [])]
            display_topics = _display_topics(semantics)
            semantic_tickers = [str(item).strip().upper() for item in (semantics.tickers if semantics else []) if str(item).strip()]
            matched_portfolio_tickers = sorted(set(semantic_tickers) & interest_context["portfolio_tickers"])
            matched_bookmark_tickers = sorted(set(semantic_tickers) & interest_context["bookmark_tickers"])
            semantic_event_type = _semantic_event_type(semantics)
            if source_filter and not any(source_filter == str(item.get("domain") or "").lower() for item in sources):
                continue
            if ticker_filter and ticker_filter not in semantic_tickers:
                continue
            if not _matches_topic_filter(topic_filter, display_topics):
                continue
            if event_type_filter and semantic_event_type != event_type_filter:
                continue
            if importance_filter and str(semantics.importance if semantics else "").strip().lower() != importance_filter:
                continue
            if scope_mode == "portfolio" and not matched_portfolio_tickers:
                continue
            if scope_mode == "bookmarks" and not matched_bookmark_tickers:
                continue
            if matches_blocked_labels(
                blocked_labels,
                article_topics=normalized_topics,
                article_tickers=semantic_tickers,
                title=article.title,
                excerpt=article.excerpt,
                content_text=article.content_text,
            ):
                continue
            payload = self._build_feed_payload(
                article,
                sources=sources,
                semantics=semantics,
                interest_context=interest_context,
                include_content=include_content,
            )
            payload["_article_updated_at"] = utc_naive_to_local_iso(article.updated_at)
            payload["_semantic_updated_at"] = utc_naive_to_local_iso(semantics.updated_at if semantics else None)
            items.append(payload)
        return items

    async def _load_related_articles(
        self,
        session,
        *,
        article: NewsArticle,
        semantics: NewsArticleSemantic | None,
        user_id: int | None,
        interest_context: dict[str, set[str]],
    ) -> list[dict[str, Any]]:
        story_key = _semantic_story_key(semantics, article)
        if not story_key:
            return []

        stmt = select(NewsArticle).where(NewsArticle.id != article.id)
        if article.published_at is not None:
            window_start = article.published_at - timedelta(days=3)
            window_end = article.published_at + timedelta(days=3)
            stmt = stmt.where(
                NewsArticle.published_at.is_not(None),
                NewsArticle.published_at >= window_start,
                NewsArticle.published_at <= window_end,
            )
        candidate_articles = (
            await session.execute(
                stmt.order_by(NewsArticle.published_at.desc().nullslast(), NewsArticle.id.desc()).limit(100)
            )
        ).scalars().all()
        if not candidate_articles:
            return []

        candidate_ids = [candidate.id for candidate in candidate_articles]
        semantic_map = await self._load_semantics_map(session, candidate_ids)
        source_map = await self._load_article_sources_map(session, candidate_ids, user_id=user_id)

        related_items: list[dict[str, Any]] = []
        for candidate in candidate_articles:
            candidate_semantics = semantic_map.get(candidate.id)
            if _semantic_story_key(candidate_semantics, candidate) != story_key:
                continue
            sources = source_map.get(candidate.id, [])
            if not sources:
                continue
            related_items.append(
                self._build_feed_payload(
                    candidate,
                    sources=sources,
                    semantics=candidate_semantics,
                    interest_context=interest_context,
                )
            )

        related_items = self._annotate_story_groups(related_items, group_by="article")
        return [
            {
                "id": int(item["id"]),
                "title": str(item["title"]),
                "published_at": item.get("published_at"),
                "canonical_url": str(item["canonical_url"]),
                "source_title": (item.get("source_labels") or [None])[0],
            }
            for item in related_items[:6]
        ]

    def _normalize_discussion_messages(self, messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in messages or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = _truncate_text(item.get("content"), limit=DISCUSSION_MAX_MESSAGE_CHARS)
            if not content:
                continue
            normalized.append({"role": role, "content": content})
        if len(normalized) > DISCUSSION_MAX_MESSAGES:
            normalized = normalized[-DISCUSSION_MAX_MESSAGES:]
        return normalized

    def _discussion_article_context(
        self,
        article: NewsArticle,
        semantics: NewsArticleSemantic | None,
        *,
        content_repaired: bool,
    ) -> dict[str, Any]:
        return {
            "title": article.title,
            "excerpt": _truncate_text(article.excerpt, limit=DISCUSSION_MAX_SNIPPET_CHARS) or None,
            "content_text": _truncate_text(article.content_text, limit=DISCUSSION_MAX_ARTICLE_CHARS) or None,
            "canonical_url": article.canonical_url,
            "language": article.language,
            "published_at": article.published_at.isoformat() if article.published_at else None,
            "topics": _display_topics(semantics),
            "tickers": [str(item) for item in (semantics.tickers if semantics else [])],
            "sectors": [str(item) for item in (semantics.sectors if semantics else [])],
            "event_labels": _semantic_event_labels(semantics),
            "content_repaired": content_repaired,
        }

    def _article_citation(self, article: NewsArticle) -> dict[str, Any]:
        return {
            "source_id": "article:primary",
            "source_type": "article",
            "title": article.title,
            "url": article.canonical_url,
            "domain": extract_domain(article.canonical_url),
            "snippet": _truncate_text(article.excerpt or article.content_text or article.title, limit=DISCUSSION_MAX_SNIPPET_CHARS),
        }

    def _discussion_search_query(
        self,
        article: NewsArticle,
        semantics: NewsArticleSemantic | None,
        messages: list[dict[str, str]],
        *,
        override: str | None,
    ) -> str:
        queries = self._heuristic_discussion_search_queries(
            article,
            semantics,
            messages,
            override=override,
        )
        return queries[0] if queries else _truncate_text(article.title, limit=240)

    def _discussion_query_subject(self, latest_user_message: str) -> str | None:
        normalized_message = _compact_whitespace(latest_user_message).strip(" ?!.,;:")
        if not normalized_message:
            return None

        folded_message = _ascii_fold(normalized_message).lower()
        for marker in (" ve ", " about "):
            index = folded_message.rfind(marker)
            if index != -1:
                subject = normalized_message[index + len(marker) :].strip(" ?!.,;:")
                return _truncate_text(subject, limit=120) or None

        for prefix in ("who is ", "what is ", "introduce ", "gioi thieu ", "giới thiệu "):
            if folded_message.startswith(_ascii_fold(prefix).lower()):
                subject = normalized_message[len(prefix) :].strip(" ?!.,;:")
                return _truncate_text(subject, limit=120) or None
        return None

    def _discussion_search_intent(self, latest_user_message: str) -> str:
        lowered = _normalized_search_text(latest_user_message)
        if any(keyword in lowered for keyword in DISCUSSION_RECAP_KEYWORDS):
            return "recap"
        if any(keyword in lowered for keyword in ("gioi thieu", "giới thiệu", "la gi", "là gì", "about", "overview", "who is", "what is")):
            return "overview"
        if any(keyword in lowered for keyword in ("thuoc ai", "thuộc ai", "owner", "ownership", "co dong", "cổ đông", "parent company")):
            return "ownership"
        if any(keyword in lowered for keyword in ("san pham", "sản phẩm", "dich vu", "dịch vụ", "hoat dong", "hoạt động", "business model", "product", "service")):
            return "business"
        if any(keyword in lowered for keyword in ("ket qua kinh doanh", "kết quả kinh doanh", "doanh thu", "loi nhuan", "lợi nhuận", "profit", "revenue", "financial")):
            return "financials"
        if any(keyword in lowered for keyword in ("so voi", "so với", "compare", "comparison", "khac gi", "khác gì")):
            return "comparison"
        if any(keyword in lowered for keyword in DISCUSSION_LATEST_KEYWORDS):
            return "latest"
        if any(keyword in lowered for keyword in DISCUSSION_EVENT_CAUSAL_KEYWORDS):
            return "event"
        return "generic"

    def _discussion_article_content_strength(self, article_context: dict[str, Any]) -> str:
        content_text = _compact_whitespace(article_context.get("content_text"))
        excerpt = _compact_whitespace(article_context.get("excerpt"))
        if len(content_text) >= 500:
            return "strong"
        if len(content_text) >= 180 or len(excerpt) >= 120:
            return "medium"
        return "weak"

    def _should_auto_search_discussion(
        self,
        *,
        intent: str,
        article_context: dict[str, Any],
        latest_user_message: str,
    ) -> bool:
        del latest_user_message
        content_strength = self._discussion_article_content_strength(article_context)
        if intent in DISCUSSION_BACKGROUND_INTENTS or intent == "latest":
            return True
        if intent == "recap":
            return False
        if intent == "event":
            return content_strength == "weak"
        return content_strength == "weak"

    def _minimum_web_citations_for_discussion(
        self,
        *,
        intent: str,
        available_web_evidence: list[dict[str, Any]],
    ) -> int:
        if intent not in DISCUSSION_BACKGROUND_INTENTS:
            return 0
        return 2 if len(available_web_evidence) >= 2 else 0

    async def _resolve_auto_discussion_search(
        self,
        *,
        article_context: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> tuple[str, bool]:
        latest_user_message = next((item["content"] for item in reversed(messages) if item["role"] == "user"), "")
        article_content_strength = self._discussion_article_content_strength(article_context)
        llm_decision = await decide_discussion_search(
            article_context=article_context,
            messages=messages,
            article_content_strength=article_content_strength,
        )
        if llm_decision:
            return (
                str(llm_decision.get("intent") or "generic"),
                bool(llm_decision.get("needs_web_search")),
            )
        fallback_intent = self._discussion_search_intent(latest_user_message)
        fallback_search = self._should_auto_search_discussion(
            intent=fallback_intent,
            article_context=article_context,
            latest_user_message=latest_user_message,
        )
        return fallback_intent, fallback_search

    def _heuristic_discussion_search_queries(
        self,
        article: NewsArticle,
        semantics: NewsArticleSemantic | None,
        messages: list[dict[str, str]],
        *,
        override: str | None,
    ) -> list[str]:
        custom_query = _truncate_text(override, limit=240)
        if custom_query:
            return [custom_query]
        latest_user_message = next((item["content"] for item in reversed(messages) if item["role"] == "user"), "")
        tickers = " ".join(
            sorted({str(item).strip().upper() for item in (semantics.tickers if semantics else []) if str(item).strip()})[:3]
        )
        subject = self._discussion_query_subject(latest_user_message)
        intent = self._discussion_search_intent(latest_user_message)

        queries: list[str] = []
        if subject:
            queries.extend(
                [
                    _truncate_text(f"\"{subject}\" Việt Nam giới thiệu công ty hoạt động sản phẩm", limit=240),
                    _truncate_text(f"\"{subject}\" Vietnam company overview products", limit=240),
                    _truncate_text(subject, limit=240),
                ]
            )
            if intent == "ownership":
                queries.append(_truncate_text(f"\"{subject}\" thuộc ai owner ownership", limit=240))
            if intent == "business":
                queries.append(_truncate_text(f"\"{subject}\" sản phẩm dịch vụ business model", limit=240))
            if intent == "financials":
                queries.append(_truncate_text(f"\"{subject}\" kết quả kinh doanh revenue profit", limit=240))
            if intent == "latest":
                queries.append(_truncate_text(f"\"{subject}\" tin mới nhất latest news", limit=240))

        message_query = _truncate_text(" ".join(part for part in [latest_user_message, tickers] if part), limit=240)
        article_query = _truncate_text(" ".join(part for part in [article.title, tickers, latest_user_message] if part), limit=240)
        article_title_query = _truncate_text(article.title, limit=240)
        queries.extend([message_query, article_query, article_title_query])

        deduped_queries: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized_query = _compact_whitespace(query)
            if not normalized_query:
                continue
            dedupe_key = normalized_query.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped_queries.append(normalized_query)
            if len(deduped_queries) >= DISCUSSION_WEB_QUERY_LIMIT:
                break
        return deduped_queries or [article_title_query]

    def _discussion_retry_queries(
        self,
        *,
        intent: str,
        subject: str | None,
        latest_user_message: str,
    ) -> list[str]:
        if not subject:
            return []

        query_pool: list[str] = []
        if intent == "overview":
            query_pool.extend(
                [
                    f"\"{subject}\" company overview official",
                    f"\"{subject}\" giới thiệu công ty",
                ]
            )
        elif intent == "ownership":
            query_pool.extend(
                [
                    f"\"{subject}\" owner parent company",
                    f"\"{subject}\" thuộc ai cổ đông",
                ]
            )
        elif intent == "business":
            query_pool.extend(
                [
                    f"\"{subject}\" product service business model",
                    f"\"{subject}\" hoạt động sản phẩm dịch vụ",
                ]
            )
        elif intent == "financials":
            query_pool.extend(
                [
                    f"\"{subject}\" financial results revenue profit",
                    f"\"{subject}\" kết quả kinh doanh lợi nhuận doanh thu",
                ]
            )
        elif intent == "comparison":
            query_pool.extend(
                [
                    latest_user_message,
                    f"\"{subject}\" comparison overview",
                ]
            )
        else:
            query_pool.extend(
                [
                    f"\"{subject}\" overview",
                    f"\"{subject}\" {latest_user_message}",
                ]
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for query in query_pool:
            normalized_query = _truncate_text(query, limit=240)
            if not normalized_query:
                continue
            key = normalized_query.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized_query)
            if len(deduped) >= DISCUSSION_WEB_RETRY_QUERY_LIMIT:
                break
        return deduped

    def _discussion_result_text(self, *, title: str | None, snippet: str | None, url: str | None) -> str:
        return " ".join(part for part in [title or "", snippet or "", url or ""] if part).strip()

    def _is_profile_like_result(self, *, title: str | None, snippet: str | None, url: str | None) -> bool:
        lowered = _normalized_search_text(self._discussion_result_text(title=title, snippet=snippet, url=url))
        return any(keyword in lowered for keyword in DISCUSSION_PROFILE_KEYWORDS)

    def _discussion_overlap_with_article(self, article: NewsArticle, *, title: str | None, snippet: str | None) -> int:
        article_tokens = _significant_search_tokens(article.title)
        result_tokens = _significant_search_tokens(self._discussion_result_text(title=title, snippet=snippet, url=None))
        return len(article_tokens.intersection(result_tokens))

    def _score_discussion_search_result(
        self,
        result: Any,
        *,
        article: NewsArticle,
        intent: str,
        subject: str | None,
    ) -> int:
        title = str(getattr(result, "title", "") or "")
        snippet = str(getattr(result, "snippet", "") or "")
        url = str(getattr(result, "url", "") or "")
        lowered = _normalized_search_text(self._discussion_result_text(title=title, snippet=snippet, url=url))

        score = 0
        if subject:
            lowered_subject = _normalized_search_text(subject)
            if lowered_subject and lowered_subject in lowered:
                score += 35

        overlap = self._discussion_overlap_with_article(article, title=title, snippet=snippet)
        if intent == "event":
            score += overlap * 8
        elif intent in DISCUSSION_BACKGROUND_INTENTS:
            score -= max(0, overlap - 1) * 8
        else:
            score += overlap * 3

        if self._is_profile_like_result(title=title, snippet=snippet, url=url):
            score += 30 if intent in {"overview", "business"} else 10
        if any(keyword in lowered for keyword in DISCUSSION_OWNERSHIP_KEYWORDS):
            score += 35 if intent == "ownership" else 5
        if any(keyword in lowered for keyword in DISCUSSION_BUSINESS_KEYWORDS):
            score += 35 if intent == "business" else 8
        if any(keyword in lowered for keyword in DISCUSSION_FINANCIAL_KEYWORDS):
            score += 35 if intent == "financials" else 8
        if any(keyword in lowered for keyword in DISCUSSION_LATEST_KEYWORDS):
            score += 20 if intent == "latest" else 0
        if len(_compact_whitespace(snippet)) < 60:
            score -= 10
        return score

    def _rank_discussion_search_results(
        self,
        results: list[Any],
        *,
        article: NewsArticle,
        intent: str,
        subject: str | None,
    ) -> list[Any]:
        return sorted(
            results,
            key=lambda result: (
                self._score_discussion_search_result(
                    result,
                    article=article,
                    intent=intent,
                    subject=subject,
                ),
                len(_compact_whitespace(str(getattr(result, "snippet", "") or ""))),
            ),
            reverse=True,
        )

    def _should_retry_discussion_search(
        self,
        ranked_results: list[Any],
        *,
        article: NewsArticle,
        intent: str,
        subject: str | None,
    ) -> bool:
        if not ranked_results:
            return True
        if intent not in DISCUSSION_BACKGROUND_INTENTS or not subject:
            return False

        top_results = ranked_results[:3]
        for result in top_results:
            score = self._score_discussion_search_result(
                result,
                article=article,
                intent=intent,
                subject=subject,
            )
            if score >= DISCUSSION_RESULT_MIN_SCORE_FOR_BACKGROUND and self._is_profile_like_result(
                title=getattr(result, "title", None),
                snippet=getattr(result, "snippet", None),
                url=getattr(result, "url", None),
            ):
                return False
        return True

    async def _collect_discussion_search_results(
        self,
        provider,
        queries: list[str],
        *,
        seen_urls: set[str],
        limit: int,
    ) -> list[Any]:
        unique_results: list[Any] = []
        for query in queries:
            query_results = await provider.search(query, limit=DISCUSSION_WEB_RESULT_LIMIT)
            for result in query_results:
                url = str(getattr(result, "url", "")).strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                unique_results.append(result)
                if len(unique_results) >= limit:
                    return unique_results
        return unique_results

    async def _discussion_search_queries(
        self,
        article: NewsArticle,
        semantics: NewsArticleSemantic | None,
        messages: list[dict[str, str]],
        *,
        override: str | None,
        article_context: dict[str, Any],
    ) -> list[str]:
        fallback_queries = self._heuristic_discussion_search_queries(
            article,
            semantics,
            messages,
            override=override,
        )
        if override:
            return fallback_queries

        llm_queries = await generate_discussion_search_queries(
            article_context=article_context,
            messages=messages,
            fallback_queries=fallback_queries,
        )
        merged_queries: list[str] = []
        seen: set[str] = set()
        for query in [*(llm_queries or []), *fallback_queries]:
            normalized_query = _compact_whitespace(query)
            if not normalized_query:
                continue
            dedupe_key = normalized_query.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged_queries.append(normalized_query)
            if len(merged_queries) >= DISCUSSION_WEB_QUERY_LIMIT:
                break
        return merged_queries or fallback_queries

    async def _build_web_discussion_evidence(
        self,
        article: NewsArticle,
        semantics: NewsArticleSemantic | None,
        messages: list[dict[str, str]],
        *,
        search_query_override: str | None,
        article_context: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        provider = get_news_search_provider(self._client)
        if provider is None:
            raise ValueError("Web search is not configured for article discussion")

        latest_user_message = next((item["content"] for item in reversed(messages) if item["role"] == "user"), "")
        intent = self._discussion_search_intent(latest_user_message)
        subject = self._discussion_query_subject(latest_user_message)
        queries = await self._discussion_search_queries(
            article,
            semantics,
            messages,
            override=search_query_override,
            article_context=article_context,
        )
        seen_urls: set[str] = {article.canonical_url}
        unique_results = await self._collect_discussion_search_results(
            provider,
            queries,
            seen_urls=seen_urls,
            limit=DISCUSSION_WEB_UNIQUE_RESULT_LIMIT,
        )

        ranked_results = self._rank_discussion_search_results(
            unique_results,
            article=article,
            intent=intent,
            subject=subject,
        )
        if self._should_retry_discussion_search(
            ranked_results,
            article=article,
            intent=intent,
            subject=subject,
        ):
            retry_queries = self._discussion_retry_queries(
                intent=intent,
                subject=subject,
                latest_user_message=latest_user_message,
            )
            retry_results = await self._collect_discussion_search_results(
                provider,
                retry_queries,
                seen_urls=seen_urls,
                limit=max(0, DISCUSSION_WEB_UNIQUE_RESULT_LIMIT - len(unique_results)),
            )
            ranked_results = self._rank_discussion_search_results(
                [*unique_results, *retry_results],
                article=article,
                intent=intent,
                subject=subject,
            )

        if not ranked_results:
            return [], 0

        evidence_items: list[dict[str, Any]] = []
        for result in ranked_results:
            snippet = _truncate_text(result.snippet, limit=DISCUSSION_MAX_SNIPPET_CHARS)
            try:
                html_text = await fetch_text(self._client, result.url)
                extracted = extract_article_payload(result.url, html_text)
                snippet = _truncate_text(
                    extracted.get("excerpt") or extracted.get("content_text") or result.snippet,
                    limit=DISCUSSION_MAX_SNIPPET_CHARS,
                )
                title = _truncate_text(extracted.get("title") or result.title, limit=220) or result.title
            except Exception:
                title = result.title
            if not snippet:
                continue
            evidence_items.append(
                {
                    "source_id": f"web:{len(evidence_items) + 1}",
                    "source_type": "web",
                    "title": title,
                    "url": result.url,
                    "domain": result.domain or extract_domain(result.url),
                    "snippet": snippet,
                }
            )
            if len(evidence_items) >= DISCUSSION_WEB_EVIDENCE_LIMIT:
                break
        return evidence_items, len(ranked_results)

    async def discuss_article(
        self,
        article_id: int,
        *,
        user_id: int,
        messages: list[dict[str, Any]] | None,
        search_mode: str = "off",
        search_query_override: str | None = None,
    ) -> dict[str, Any] | None:
        normalized_messages = self._normalize_discussion_messages(messages)
        if not normalized_messages:
            raise ValueError("Discussion requires at least one message")

        async with async_session() as session:
            article = await session.get(NewsArticle, article_id)
            if article is None:
                return None
            sources = await self._load_article_sources(session, article_id, user_id=user_id)
            if not sources:
                return None

            repaired = False
            if not (article.content_text or "").strip():
                repaired = await self._repair_article_content_if_needed(session, article, force_refresh=True)
                if repaired:
                    await self._upsert_article_semantics(session, article)
                    article.llm_summary = None
                    await session.commit()
                    await session.refresh(article)

            semantic_map = await self._load_semantics_map(session, [article_id])
            semantics = semantic_map.get(article_id)

        article_context = self._discussion_article_context(
            article,
            semantics,
            content_repaired=repaired,
        )
        latest_user_message = next((item["content"] for item in reversed(normalized_messages) if item["role"] == "user"), "")
        intent = self._discussion_search_intent(latest_user_message)
        effective_search_mode = "off"
        if search_mode == "on":
            effective_search_mode = "on"
        elif search_mode == "auto":
            intent, should_search = await self._resolve_auto_discussion_search(
                article_context=article_context,
                messages=normalized_messages,
            )
            if should_search:
                effective_search_mode = "on"

        evidence_items = [self._article_citation(article)]
        warning_parts: list[str] = []
        if not article_context.get("content_text"):
            warning_parts.append("Full article body was unavailable, so the answer is grounded in the title and excerpt.")

        web_result_count = 0
        web_evidence: list[dict[str, Any]] = []
        if effective_search_mode == "on":
            web_evidence, web_result_count = await self._build_web_discussion_evidence(
                article,
                semantics,
                normalized_messages,
                search_query_override=search_query_override,
                article_context=article_context,
            )
            evidence_items.extend(web_evidence)
            if web_result_count == 0:
                warning_parts.append("No relevant web results were found for this discussion.")
            elif not web_evidence:
                warning_parts.append("Web search found results, but readable supporting material could not be extracted.")

        llm_payload = await discuss_article_with_context(
            article_context=article_context,
            messages=normalized_messages,
            evidence_items=evidence_items,
            search_web=effective_search_mode == "on",
        )
        if not llm_payload:
            raise ValueError("Unable to generate discussion response for this article")

        evidence_by_id = {
            str(item["source_id"]): item
            for item in evidence_items
            if isinstance(item, dict) and item.get("source_id")
        }
        minimum_web_citations = self._minimum_web_citations_for_discussion(
            intent=intent,
            available_web_evidence=web_evidence,
        )
        citation_ids = [
            citation_id
            for citation_id in llm_payload.get("cited_source_ids", [])
            if citation_id in evidence_by_id
        ]
        if minimum_web_citations > 1:
            cited_web_ids = [
                citation_id
                for citation_id in citation_ids
                if evidence_by_id[citation_id].get("source_type") == "web"
            ]
            if 0 < len(cited_web_ids) < minimum_web_citations:
                for item in web_evidence:
                    source_id = str(item.get("source_id") or "")
                    if not source_id or source_id in cited_web_ids:
                        continue
                    citation_ids.append(source_id)
                    cited_web_ids.append(source_id)
                    if len(cited_web_ids) >= minimum_web_citations:
                        warning_parts.append(
                            "Additional web citations were attached because multiple outside sources were available for this question."
                        )
                        break
        if not citation_ids:
            citation_ids = ["article:primary"]
            if effective_search_mode == "on" and web_result_count > 0:
                warning_parts.append("Fallback article citation attached because the model did not return explicit source references.")

        citations = [evidence_by_id[citation_id] for citation_id in citation_ids]
        used_web_search = any(item.get("source_type") == "web" for item in citations)
        llm_warning = _compact_whitespace(llm_payload.get("warning"))
        if llm_warning:
            warning_parts.append(llm_warning)
        if effective_search_mode == "on" and web_result_count > 0 and not used_web_search:
            warning_parts.append("Web search ran, but the answer stayed grounded in the article because outside evidence was not selected.")

        deduped_warning_parts: list[str] = []
        seen_warning_parts: set[str] = set()
        for part in warning_parts:
            normalized_part = _compact_whitespace(part)
            if not normalized_part or normalized_part in seen_warning_parts:
                continue
            seen_warning_parts.add(normalized_part)
            deduped_warning_parts.append(normalized_part)

        return {
            "assistant_message": str(llm_payload["assistant_message"]),
            "citations": citations,
            "search_mode": search_mode if search_mode in {"off", "auto", "on"} else "off",
            "effective_search_mode": effective_search_mode,
            "used_web_search": used_web_search,
            "web_results_count": web_result_count,
            "warning": " ".join(deduped_warning_parts) if deduped_warning_parts else None,
        }

    async def get_feed(
        self,
        *,
        user_id: int | None,
        source: str | None = None,
        ticker: str | None = None,
        topic: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        cursor: str | None = None,
        limit: int = 20,
        sort: str = "latest",
        scope: str = "all",
        bookmark_group_id: int | None = None,
        event_type: str | None = None,
        importance: str | None = None,
        group_by: str = "article",
    ) -> dict[str, Any]:
        safe_limit = min(50, max(1, limit))
        offset = self._parse_offset_cursor(cursor)
        scope_mode = scope if user_id is not None and scope in {"all", "portfolio", "bookmarks"} else "all"
        sort_mode = sort if user_id is not None and sort == "relevance" else "latest"
        group_mode = "story" if group_by == "story" else "article"
        async with async_session() as session:
            items = await self._collect_feed_items(
                session,
                user_id=user_id,
                source=source,
                ticker=ticker,
                topic=topic,
                date_from=date_from,
                date_to=date_to,
                scope=scope_mode,
                bookmark_group_id=bookmark_group_id,
                event_type=event_type,
                importance=importance,
            )
            items.sort(
                key=lambda item: (
                    self._story_sort_key(item) if sort_mode == "relevance" else (0, str(item.get("published_at") or ""), int(item.get("id") or 0))
                ),
                reverse=True,
            )
            grouped_items = self._annotate_story_groups(items, group_by=group_mode)
            total_count = len(grouped_items)
            paged_items = grouped_items[offset : offset + safe_limit]
            next_cursor = None
            if offset + safe_limit < total_count:
                next_cursor = f"offset:{offset + safe_limit}"
            return {"items": paged_items, "count": total_count, "next_cursor": next_cursor}

    async def get_quick_glance_digest(
        self,
        *,
        user_id: int | None,
        window_hours: int,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        now = utc_now()
        window_start = now - timedelta(hours=window_hours)
        async with async_session() as session:
            items = await self._collect_feed_items(
                session,
                user_id=user_id,
                date_from=window_start,
                date_to=now,
                include_content=True,
            )
            if not items:
                return {
                    "window_hours": window_hours,
                    "article_count": 0,
                    "oldest_article_at": None,
                    "newest_article_at": None,
                    "generated_at": utc_naive_to_local_iso(now),
                    "cache_hit": False,
                    "summary": None,
                    "highlights": [],
                    "key_articles": [],
                }

            newest_article_at = max(str(item.get("published_at") or "") for item in items) or None
            oldest_article_at = min(str(item.get("published_at") or "") for item in items) or None
            ranked_items = sorted(items, key=self._story_sort_key, reverse=True)
            story_items = self._annotate_story_groups(ranked_items, group_by="story")
            evidence_story_items = self._select_quick_glance_evidence_story_items(story_items)
            evidence_items = [self._build_quick_glance_evidence_item(item) for item in evidence_story_items]
            evidence_fingerprint = self._build_quick_glance_fingerprint(items, evidence_items)
            viewer_key = self._quick_glance_viewer_key(user_id)

            latest_cached_digest = (
                await session.execute(
                    select(NewsQuickGlanceDigest)
                    .where(
                        NewsQuickGlanceDigest.viewer_key == viewer_key,
                        NewsQuickGlanceDigest.window_hours == window_hours,
                    )
                    .order_by(NewsQuickGlanceDigest.generated_at.desc(), NewsQuickGlanceDigest.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            cached_digest = (
                await session.execute(
                    select(NewsQuickGlanceDigest)
                    .where(
                        NewsQuickGlanceDigest.viewer_key == viewer_key,
                        NewsQuickGlanceDigest.window_hours == window_hours,
                        NewsQuickGlanceDigest.evidence_fingerprint == evidence_fingerprint,
                    )
                    .order_by(NewsQuickGlanceDigest.generated_at.desc(), NewsQuickGlanceDigest.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if cached_digest is not None and not force_refresh:
                return self._hydrate_quick_glance_payload(cached_digest, cache_hit=True)
            if (
                latest_cached_digest is not None
                and not force_refresh
                and not self._should_regenerate_quick_glance_digest(
                    now=now,
                    cached_digest=latest_cached_digest,
                    items=items,
                )
            ):
                return self._hydrate_quick_glance_payload(latest_cached_digest, cache_hit=True)

            llm_payload = await generate_quick_glance_digest(
                window_hours=window_hours,
                article_count=len(items),
                highlights_target=QUICK_GLANCE_HIGHLIGHT_LIMIT,
                evidence_items=evidence_items,
            )
            if not llm_payload:
                raise ValueError("Unable to generate quick glance digest for this timeframe")

            key_articles = self._select_quick_glance_key_articles(
                story_items,
                highlighted_article_ids=[
                    article_id
                    for highlight in llm_payload.get("highlights", [])
                    for article_id in highlight.get("article_ids", [])
                ],
            )
            payload = {
                "window_hours": window_hours,
                "article_count": len(items),
                "oldest_article_at": oldest_article_at,
                "newest_article_at": newest_article_at,
                "summary": llm_payload["summary"],
                "highlights": llm_payload.get("highlights", []),
                "key_articles": key_articles,
            }
            if cached_digest is not None:
                cache_row = cached_digest
                cache_row.payload = payload
                cache_row.generated_at = now
            else:
                cache_row = NewsQuickGlanceDigest(
                    user_id=user_id,
                    viewer_key=viewer_key,
                    window_hours=window_hours,
                    evidence_fingerprint=evidence_fingerprint,
                    payload=payload,
                    generated_at=now,
                )
                session.add(cache_row)
            await session.commit()
            await session.refresh(cache_row)
            return self._hydrate_quick_glance_payload(cache_row, cache_hit=False)

    async def get_article_detail(self, article_id: int, *, user_id: int | None) -> dict[str, Any] | None:
        async with async_session() as session:
            article = await session.get(NewsArticle, article_id)
            if article is None:
                return None
            sources = await self._load_article_sources(session, article_id, user_id=user_id)
            if not sources:
                return None
            repaired = await self._repair_article_content_if_needed(session, article)
            if repaired:
                await self._upsert_article_semantics(session, article)
                article.llm_summary = None
                await session.commit()
                await session.refresh(article)
            semantic_map = await self._load_semantics_map(session, [article_id])
            interest_context = await self._load_user_interest_context(session, user_id=user_id)
            semantics = semantic_map.get(article_id)
            payload = self._build_feed_payload(
                article,
                sources=sources,
                semantics=semantics,
                interest_context=interest_context,
                include_content=True,
            )
            related_articles = await self._load_related_articles(
                session,
                article=article,
                semantics=semantics,
                user_id=user_id,
                interest_context=interest_context,
            )
            payload["related_article_ids"] = [int(item["id"]) for item in related_articles]
            payload["related_articles"] = related_articles
            return payload

    async def refresh_article_content(self, article_id: int, *, user_id: int | None) -> dict[str, Any] | None:
        async with async_session() as session:
            article = await session.get(NewsArticle, article_id)
            if article is None:
                return None
            sources = await self._load_article_sources(session, article_id, user_id=user_id)
            if not sources:
                return None
            repaired = await self._repair_article_content_if_needed(session, article, force_refresh=True)
            await self._upsert_article_semantics(session, article)
            if repaired:
                article.llm_summary = None
                await session.commit()
                await session.refresh(article)
            if not repaired and not article.content_text:
                raise ValueError("Unable to refresh article content")
            semantic_map = await self._load_semantics_map(session, [article_id])
            interest_context = await self._load_user_interest_context(session, user_id=user_id)
            semantics = semantic_map.get(article_id)
            payload = self._build_feed_payload(
                article,
                sources=sources,
                semantics=semantics,
                interest_context=interest_context,
                include_content=True,
            )
            related_articles = await self._load_related_articles(
                session,
                article=article,
                semantics=semantics,
                user_id=user_id,
                interest_context=interest_context,
            )
            payload["related_article_ids"] = [int(item["id"]) for item in related_articles]
            payload["related_articles"] = related_articles
            return payload

    async def generate_article_summary(
        self,
        article_id: int,
        *,
        user_id: int | None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        async with async_session() as session:
            article = await session.get(NewsArticle, article_id)
            if article is None:
                return None
            sources = await self._load_article_sources(session, article_id, user_id=user_id)
            if not sources:
                return None
            if force_refresh or not article.llm_summary:
                summary = await summarize_article(
                    article.title,
                    article.excerpt,
                    article.content_text,
                    language=article.language,
                )
                if not summary:
                    raise ValueError("Unable to generate LLM summary for this article")
                article.llm_summary = summary
                await session.commit()
                await session.refresh(article)
            semantic_map = await self._load_semantics_map(session, [article_id])
            interest_context = await self._load_user_interest_context(session, user_id=user_id)
            payload = self._build_feed_payload(
                article,
                sources=sources,
                semantics=semantic_map.get(article_id),
                interest_context=interest_context,
            )
            payload["related_article_ids"] = []
            return payload

    async def _run_loop(self) -> None:
        while True:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(max(5.0, float(settings.news_poll_interval_seconds)))

    async def _poll_once(self) -> None:
        async with async_session() as session:
            due_feeds = (
                await session.execute(
                    select(NewsSiteFeed)
                    .outerjoin(
                        NewsSourceSubscription,
                        and_(
                            NewsSourceSubscription.site_feed_id == NewsSiteFeed.id,
                            NewsSourceSubscription.enabled.is_(True),
                        ),
                    )
                    .where(
                        NewsSiteFeed.validation_status == "valid",
                        NewsSiteFeed.enabled.is_(True),
                        or_(NewsSiteFeed.is_public.is_(True), NewsSourceSubscription.id.is_not(None)),
                        or_(NewsSiteFeed.next_poll_at.is_(None), NewsSiteFeed.next_poll_at <= utc_now()),
                    )
                    .order_by(NewsSiteFeed.next_poll_at.asc().nullsfirst(), NewsSiteFeed.id.asc())
                    .limit(max(1, int(settings.news_ingestion_batch_size)))
                )
            ).scalars().unique().all()
            due_crawls = (
                await session.execute(
                    select(NewsCrawlSource)
                    .outerjoin(
                        NewsSourceSubscription,
                        and_(
                            NewsSourceSubscription.crawl_source_id == NewsCrawlSource.id,
                            NewsSourceSubscription.enabled.is_(True),
                        ),
                    )
                    .where(
                        NewsCrawlSource.validation_status == "valid",
                        NewsCrawlSource.enabled.is_(True),
                        or_(NewsCrawlSource.is_public.is_(True), NewsSourceSubscription.id.is_not(None)),
                        or_(NewsCrawlSource.next_poll_at.is_(None), NewsCrawlSource.next_poll_at <= utc_now()),
                    )
                    .order_by(NewsCrawlSource.next_poll_at.asc().nullsfirst(), NewsCrawlSource.id.asc())
                    .limit(max(1, int(settings.news_ingestion_batch_size)))
                )
            ).scalars().unique().all()

        for feed in due_feeds:
            await self._ingest_feed(feed.id)
        for source in due_crawls:
            await self._ingest_crawl_source(source.id)

    async def _ingest_feed(self, feed_id: int) -> None:
        async with async_session() as session:
            feed = await session.get(NewsSiteFeed, feed_id)
            if feed is None:
                return
            run = NewsIngestionRun(source_type="rss", site_feed_id=feed.id, status="running", started_at=utc_now())
            session.add(run)
            await session.commit()
            await session.refresh(run)

            try:
                feed_text = await fetch_text(self._client, feed.feed_url)
                entries = parse_feed_entries(feed_text)
                stored_count = 0
                for entry in entries[:25]:
                    stored_count += await self._store_article_candidate(
                        session,
                        source_type="rss",
                        site_feed_id=feed.id,
                        crawl_source_id=None,
                        article_url=entry.link,
                        title=entry.title,
                        excerpt=entry.summary,
                        published_at=entry.published_at,
                    )
                now = utc_now()
                feed.last_polled_at = now
                feed.last_success_at = now
                feed.last_failure_at = None
                feed.validation_error = None
                feed.next_poll_at = now + timedelta(minutes=feed.poll_interval_minutes)
                run.status = "succeeded"
                run.fetched_count = len(entries[:25])
                run.stored_count = stored_count
                run.finished_at = now
                await session.commit()
            except Exception as exc:
                now = utc_now()
                feed.last_polled_at = now
                feed.last_failure_at = now
                feed.validation_error = str(exc)[:1000]
                feed.next_poll_at = now + timedelta(minutes=feed.poll_interval_minutes)
                run.status = "failed"
                run.error = str(exc)[:1000]
                run.finished_at = now
                await session.commit()

    async def _ingest_crawl_source(self, source_id: int) -> None:
        async with async_session() as session:
            source = await session.get(NewsCrawlSource, source_id)
            if source is None:
                return
            run = NewsIngestionRun(source_type="crawl", crawl_source_id=source.id, status="running", started_at=utc_now())
            session.add(run)
            await session.commit()
            await session.refresh(run)

            try:
                listing_html = await fetch_text(self._client, source.listing_url)
                article_urls = extract_links_with_selector(source.listing_url, listing_html, source.article_link_selector)
                stored_count = 0
                for article_url in article_urls[:20]:
                    html_text = await fetch_text(self._client, article_url)
                    payload = extract_article_payload(
                        article_url,
                        html_text,
                        content_selector=source.content_selector,
                        excerpt_selector=source.excerpt_selector,
                    )
                    stored_count += await self._store_article_candidate(
                        session,
                        source_type="crawl",
                        site_feed_id=None,
                        crawl_source_id=source.id,
                        article_url=payload["canonical_url"],
                        title=payload["title"],
                        excerpt=payload["excerpt"],
                        published_at=payload["published_at"],
                        content_text=payload["content_text"],
                        language=payload["language"],
                        image_url=payload["image_url"],
                    )
                now = utc_now()
                source.last_polled_at = now
                source.last_success_at = now
                source.last_failure_at = None
                source.validation_error = None
                source.next_poll_at = now + timedelta(minutes=source.poll_interval_minutes)
                run.status = "succeeded"
                run.fetched_count = len(article_urls[:20])
                run.stored_count = stored_count
                run.finished_at = now
                await session.commit()
            except Exception as exc:
                now = utc_now()
                source.last_polled_at = now
                source.last_failure_at = now
                source.validation_error = str(exc)[:1000]
                source.next_poll_at = now + timedelta(minutes=source.poll_interval_minutes)
                run.status = "failed"
                run.error = str(exc)[:1000]
                run.finished_at = now
                await session.commit()

    async def _store_article_candidate(
        self,
        session,
        *,
        source_type: str,
        site_feed_id: int | None,
        crawl_source_id: int | None,
        article_url: str,
        title: str,
        excerpt: str | None,
        published_at: datetime | None,
        content_text: str | None = None,
        language: str | None = None,
        image_url: str | None = None,
    ) -> int:
        normalized_url = normalize_url(article_url)
        article_payload = {
            "canonical_url": normalized_url,
            "title": title.strip(),
            "excerpt": excerpt.strip() if excerpt else None,
            "content_text": content_text,
            "published_at": published_at,
            "language": language,
            "image_url": image_url,
        }
        if content_text is None:
            try:
                html_text = await fetch_text(self._client, normalized_url)
                extracted = extract_article_payload(normalized_url, html_text)
                article_payload.update({key: value for key, value in extracted.items() if value is not None})
            except Exception:
                pass
        article_hash = self._content_hash(
            article_payload.get("title"),
            article_payload.get("excerpt"),
            article_payload.get("content_text"),
        )
        article_payload["content_hash"] = article_hash

        article = (
            await session.execute(select(NewsArticle).where(NewsArticle.canonical_url == article_payload["canonical_url"]))
        ).scalar_one_or_none()
        if article is None and article_hash:
            article = (
                await session.execute(select(NewsArticle).where(NewsArticle.content_hash == article_hash))
            ).scalar_one_or_none()

        should_classify = False
        if article is None:
            article = NewsArticle(**article_payload)
            session.add(article)
            await session.flush()
            should_classify = True
        else:
            previous_hash = article.content_hash
            for key, value in article_payload.items():
                if value is not None:
                    setattr(article, key, value)
            should_classify = previous_hash != article.content_hash

        mapping = await self._get_or_create_article_source(
            session,
            article_id=article.id,
            site_feed_id=site_feed_id,
            crawl_source_id=crawl_source_id,
            article_url=normalized_url,
        )
        if mapping and should_classify:
            await self._upsert_article_semantics(session, article)

        await session.flush()
        return 1

    async def _get_or_create_site(
        self,
        session,
        *,
        homepage_url: str,
        display_name: str | None,
        is_public: bool,
    ) -> NewsSite:
        domain = extract_domain(homepage_url)
        site = (
            await session.execute(
                select(NewsSite).where(NewsSite.domain == domain, NewsSite.homepage_url == homepage_url)
            )
        ).scalar_one_or_none()
        if site is None:
            site = NewsSite(
                domain=domain,
                homepage_url=homepage_url,
                display_name=display_name,
                is_public=is_public,
            )
            session.add(site)
            await session.flush()
        else:
            site.is_public = site.is_public or is_public
            if display_name and not site.display_name:
                site.display_name = display_name
        return site

    async def _get_or_create_feed(
        self,
        session,
        *,
        site_id: int,
        feed_url: str,
        title: str | None,
        kind: str,
        discovery_method: str,
        validation_status: str,
        is_public: bool,
        poll_interval_minutes: int,
    ) -> NewsSiteFeed:
        feed = (
            await session.execute(select(NewsSiteFeed).where(NewsSiteFeed.feed_url == feed_url))
        ).scalar_one_or_none()
        if feed is None:
            feed = NewsSiteFeed(
                site_id=site_id,
                feed_url=feed_url,
                title=title,
                kind=kind,
                discovery_method=discovery_method,
                validation_status=validation_status,
                is_public=is_public,
                poll_interval_minutes=poll_interval_minutes,
                next_poll_at=utc_now(),
            )
            session.add(feed)
            await session.flush()
        else:
            feed.site_id = site_id
            feed.title = title or feed.title
            feed.kind = kind or feed.kind
            feed.discovery_method = discovery_method or feed.discovery_method
            feed.validation_status = validation_status or feed.validation_status
            feed.is_public = feed.is_public or is_public
            feed.poll_interval_minutes = poll_interval_minutes or feed.poll_interval_minutes
            if feed.next_poll_at is None:
                feed.next_poll_at = utc_now()
        return feed

    async def _get_or_create_crawl_source(
        self,
        session,
        *,
        site_id: int,
        listing_url: str,
        article_link_selector: str,
        content_selector: str,
        excerpt_selector: str | None,
        pagination_config: dict[str, Any] | None,
        validation_status: str,
        is_public: bool,
        poll_interval_minutes: int,
    ) -> NewsCrawlSource:
        source = (
            await session.execute(
                select(NewsCrawlSource).where(
                    NewsCrawlSource.listing_url == listing_url,
                    NewsCrawlSource.article_link_selector == article_link_selector,
                )
            )
        ).scalar_one_or_none()
        if source is None:
            source = NewsCrawlSource(
                site_id=site_id,
                listing_url=listing_url,
                article_link_selector=article_link_selector,
                content_selector=content_selector,
                excerpt_selector=excerpt_selector,
                pagination_config=pagination_config,
                validation_status=validation_status,
                is_public=is_public,
                poll_interval_minutes=poll_interval_minutes,
                next_poll_at=utc_now(),
            )
            session.add(source)
            await session.flush()
        else:
            source.site_id = site_id
            source.content_selector = content_selector
            source.excerpt_selector = excerpt_selector
            source.pagination_config = pagination_config
            source.validation_status = validation_status
            source.is_public = source.is_public or is_public
            source.poll_interval_minutes = poll_interval_minutes or source.poll_interval_minutes
            if source.next_poll_at is None:
                source.next_poll_at = utc_now()
        return source

    async def _upsert_subscription(
        self,
        session,
        *,
        user_id: int,
        site_feed_id: int | None,
        crawl_source_id: int | None,
    ) -> NewsSourceSubscription:
        subscription = await self._get_existing_subscription(
            session,
            user_id=user_id,
            site_feed_id=site_feed_id,
            crawl_source_id=crawl_source_id,
        )
        if subscription is None:
            subscription = NewsSourceSubscription(
                user_id=user_id,
                site_feed_id=site_feed_id,
                crawl_source_id=crawl_source_id,
                enabled=True,
            )
            session.add(subscription)
            await session.flush()
        else:
            subscription.enabled = True
        return subscription

    async def _maybe_delete_orphaned_private_source(self, session, *, source_type: str, source_id: int) -> None:
        if source_type == "rss":
            remaining_subscription = (
                await session.execute(
                    select(NewsSourceSubscription.id).where(NewsSourceSubscription.site_feed_id == source_id).limit(1)
                )
            ).scalar_one_or_none()
            if remaining_subscription is None:
                await self._hard_delete_source(session, source_type=source_type, source_id=source_id)
            return

        if source_type == "crawl":
            remaining_subscription = (
                await session.execute(
                    select(NewsSourceSubscription.id).where(NewsSourceSubscription.crawl_source_id == source_id).limit(1)
                )
            ).scalar_one_or_none()
            if remaining_subscription is None:
                await self._hard_delete_source(session, source_type=source_type, source_id=source_id)
            return

        raise ValueError("Unsupported source type")

    async def _hard_delete_source(self, session, *, source_type: str, source_id: int) -> None:
        if source_type == "rss":
            feed = await session.get(NewsSiteFeed, source_id)
            if feed is None:
                raise ValueError("News source not found")
            site_id = int(feed.site_id)
            await session.execute(delete(NewsSourceSubscription).where(NewsSourceSubscription.site_feed_id == source_id))
            await session.execute(delete(NewsArticleSource).where(NewsArticleSource.site_feed_id == source_id))
            await session.execute(delete(NewsIngestionRun).where(NewsIngestionRun.site_feed_id == source_id))
            await session.delete(feed)
            await session.flush()
            await self._cleanup_orphaned_site(session, site_id=site_id)
            return

        if source_type == "crawl":
            source = await session.get(NewsCrawlSource, source_id)
            if source is None:
                raise ValueError("News source not found")
            site_id = int(source.site_id)
            await session.execute(delete(NewsSourceSubscription).where(NewsSourceSubscription.crawl_source_id == source_id))
            await session.execute(delete(NewsArticleSource).where(NewsArticleSource.crawl_source_id == source_id))
            await session.execute(delete(NewsIngestionRun).where(NewsIngestionRun.crawl_source_id == source_id))
            await session.delete(source)
            await session.flush()
            await self._cleanup_orphaned_site(session, site_id=site_id)
            return

        raise ValueError("Unsupported source type")

    async def _cleanup_orphaned_site(self, session, *, site_id: int) -> None:
        site = await session.get(NewsSite, site_id)
        if site is None:
            return

        has_feed = (
            await session.execute(select(NewsSiteFeed.id).where(NewsSiteFeed.site_id == site_id).limit(1))
        ).scalar_one_or_none()
        has_crawl_source = (
            await session.execute(select(NewsCrawlSource.id).where(NewsCrawlSource.site_id == site_id).limit(1))
        ).scalar_one_or_none()

        if has_feed is None and has_crawl_source is None:
            await session.delete(site)
            await session.flush()

    async def _get_existing_subscription(
        self,
        session,
        *,
        user_id: int,
        site_feed_id: int | None,
        crawl_source_id: int | None,
    ) -> NewsSourceSubscription | None:
        stmt = select(NewsSourceSubscription).where(NewsSourceSubscription.user_id == user_id)
        if site_feed_id is not None:
            stmt = stmt.where(NewsSourceSubscription.site_feed_id == site_feed_id)
        if crawl_source_id is not None:
            stmt = stmt.where(NewsSourceSubscription.crawl_source_id == crawl_source_id)
        return (await session.execute(stmt)).scalar_one_or_none()

    async def _get_subscription(
        self,
        session,
        *,
        user_id: int,
        site_feed_id: int | None = None,
        crawl_source_id: int | None = None,
    ) -> NewsSourceSubscription:
        subscription = await self._get_existing_subscription(
            session,
            user_id=user_id,
            site_feed_id=site_feed_id,
            crawl_source_id=crawl_source_id,
        )
        if subscription is None:
            raise ValueError("News source not found")
        return subscription

    async def _get_or_create_article_source(
        self,
        session,
        *,
        article_id: int,
        site_feed_id: int | None,
        crawl_source_id: int | None,
        article_url: str,
    ) -> NewsArticleSource:
        stmt = select(NewsArticleSource).where(NewsArticleSource.article_id == article_id)
        if site_feed_id is not None:
            stmt = stmt.where(NewsArticleSource.site_feed_id == site_feed_id)
        if crawl_source_id is not None:
            stmt = stmt.where(NewsArticleSource.crawl_source_id == crawl_source_id)
        mapping = (await session.execute(stmt)).scalar_one_or_none()
        if mapping is None:
            mapping = NewsArticleSource(
                article_id=article_id,
                site_feed_id=site_feed_id,
                crawl_source_id=crawl_source_id,
                article_url=article_url,
            )
            session.add(mapping)
            await session.flush()
        return mapping

    async def _get_or_create_preference(self, session, user_id: int) -> NewsUserPreference:
        preference = (
            await session.execute(select(NewsUserPreference).where(NewsUserPreference.user_id == user_id))
        ).scalar_one_or_none()
        if preference is None:
            preference = NewsUserPreference(user_id=user_id, blocked_labels=[])
            session.add(preference)
            await session.flush()
        return preference

    async def _load_blocked_labels(self, session, user_id: int | None) -> list[str]:
        if user_id is None:
            return []
        preference = (
            await session.execute(select(NewsUserPreference).where(NewsUserPreference.user_id == user_id))
        ).scalar_one_or_none()
        return [str(item) for item in (preference.blocked_labels if preference else [])]

    async def _load_semantics_map(self, session, article_ids: list[int]) -> dict[int, NewsArticleSemantic]:
        if not article_ids:
            return {}
        rows = (
            await session.execute(
                select(NewsArticleSemantic).where(NewsArticleSemantic.article_id.in_(article_ids))
            )
        ).scalars().all()
        return {row.article_id: row for row in rows}

    async def _load_article_sources(self, session, article_id: int, *, user_id: int | None) -> list[dict[str, Any]]:
        source_map = await self._load_article_sources_map(session, [article_id], user_id=user_id)
        return source_map.get(article_id, [])

    async def _load_article_sources_map(
        self,
        session,
        article_ids: list[int],
        *,
        user_id: int | None,
    ) -> dict[int, list[dict[str, Any]]]:
        if not article_ids:
            return {}

        grouped_sources: dict[int, list[dict[str, Any]]] = defaultdict(list)

        feed_stmt = (
            select(NewsArticleSource, NewsSiteFeed, NewsSite)
            .join(NewsSiteFeed, NewsSiteFeed.id == NewsArticleSource.site_feed_id)
            .join(NewsSite, NewsSite.id == NewsSiteFeed.site_id)
            .where(NewsArticleSource.article_id.in_(article_ids))
        )
        if user_id is None:
            feed_stmt = feed_stmt.where(NewsSiteFeed.is_public.is_(True))
        else:
            feed_stmt = (
                feed_stmt.outerjoin(
                    NewsSourceSubscription,
                    and_(
                        NewsSourceSubscription.site_feed_id == NewsSiteFeed.id,
                        NewsSourceSubscription.user_id == user_id,
                        NewsSourceSubscription.enabled.is_(True),
                    ),
                )
                .where(or_(NewsSiteFeed.is_public.is_(True), NewsSourceSubscription.id.is_not(None)))
            )
        feed_rows = (await session.execute(feed_stmt)).all()
        for mapping, feed, site in feed_rows:
            grouped_sources[mapping.article_id].append(
                {
                    "source_type": "rss",
                    "source_id": feed.id,
                    "label": site.domain if site else extract_domain(mapping.article_url),
                    "domain": site.domain if site else extract_domain(mapping.article_url),
                    "article_url": mapping.article_url,
                    "is_public": bool(feed.is_public),
                }
            )

        crawl_stmt = (
            select(NewsArticleSource, NewsCrawlSource, NewsSite)
            .join(NewsCrawlSource, NewsCrawlSource.id == NewsArticleSource.crawl_source_id)
            .join(NewsSite, NewsSite.id == NewsCrawlSource.site_id)
            .where(NewsArticleSource.article_id.in_(article_ids))
        )
        if user_id is None:
            crawl_stmt = crawl_stmt.where(NewsCrawlSource.is_public.is_(True))
        else:
            crawl_stmt = (
                crawl_stmt.outerjoin(
                    NewsSourceSubscription,
                    and_(
                        NewsSourceSubscription.crawl_source_id == NewsCrawlSource.id,
                        NewsSourceSubscription.user_id == user_id,
                        NewsSourceSubscription.enabled.is_(True),
                    ),
                )
                .where(or_(NewsCrawlSource.is_public.is_(True), NewsSourceSubscription.id.is_not(None)))
            )
        crawl_rows = (await session.execute(crawl_stmt)).all()
        for mapping, source, site in crawl_rows:
            grouped_sources[mapping.article_id].append(
                {
                    "source_type": "crawl",
                    "source_id": source.id,
                    "label": site.domain if site else extract_domain(mapping.article_url),
                    "domain": site.domain if site else extract_domain(mapping.article_url),
                    "article_url": mapping.article_url,
                    "is_public": bool(source.is_public),
                }
            )
        return dict(grouped_sources)

    async def _serialize_feed_source_admin(self, session, feed: NewsSiteFeed, *, user_id: int | None = None) -> dict[str, Any]:
        site = await session.get(NewsSite, feed.site_id)
        subscriptions = (
            await session.execute(
                select(NewsSourceSubscription).where(NewsSourceSubscription.site_feed_id == feed.id)
            )
        ).scalars().all()
        subscription_count = len(subscriptions)
        article_count = len(
            (
                await session.execute(
                    select(NewsArticleSource).where(NewsArticleSource.site_feed_id == feed.id)
                )
            ).scalars().all()
        )
        return {
            "kind": "rss",
            "id": feed.id,
            "site_id": site.id if site else None,
            "site_name": site.display_name if site else None,
            "site_url": site.homepage_url if site else None,
            "domain": site.domain if site else None,
            "feed_url": feed.feed_url,
            "title": feed.title,
            "discovery_method": feed.discovery_method,
            "validation_status": feed.validation_status,
            "validation_error": feed.validation_error,
            "poll_interval_minutes": feed.poll_interval_minutes,
            "last_polled_at": utc_naive_to_local_iso(feed.last_polled_at),
            "next_poll_at": utc_naive_to_local_iso(feed.next_poll_at),
            "last_success_at": utc_naive_to_local_iso(feed.last_success_at),
            "last_failure_at": utc_naive_to_local_iso(feed.last_failure_at),
            "enabled": bool(feed.enabled),
            "is_public": feed.is_public,
            "subscription_count": subscription_count,
            "article_count": article_count,
        }

    async def _serialize_crawl_source_admin(self, session, source: NewsCrawlSource, *, user_id: int | None = None) -> dict[str, Any]:
        site = await session.get(NewsSite, source.site_id)
        subscriptions = (
            await session.execute(
                select(NewsSourceSubscription).where(NewsSourceSubscription.crawl_source_id == source.id)
            )
        ).scalars().all()
        subscription_count = len(subscriptions)
        article_count = len(
            (
                await session.execute(
                    select(NewsArticleSource).where(NewsArticleSource.crawl_source_id == source.id)
                )
            ).scalars().all()
        )
        return {
            "kind": "crawl",
            "id": source.id,
            "site_id": site.id if site else None,
            "site_name": site.display_name if site else None,
            "site_url": site.homepage_url if site else None,
            "domain": site.domain if site else None,
            "listing_url": source.listing_url,
            "article_link_selector": source.article_link_selector,
            "content_selector": source.content_selector,
            "excerpt_selector": source.excerpt_selector,
            "pagination_config": source.pagination_config,
            "validation_status": source.validation_status,
            "validation_error": source.validation_error,
            "poll_interval_minutes": source.poll_interval_minutes,
            "last_polled_at": utc_naive_to_local_iso(source.last_polled_at),
            "next_poll_at": utc_naive_to_local_iso(source.next_poll_at),
            "last_success_at": utc_naive_to_local_iso(source.last_success_at),
            "last_failure_at": utc_naive_to_local_iso(source.last_failure_at),
            "enabled": bool(source.enabled),
            "is_public": source.is_public,
            "subscription_count": subscription_count,
            "article_count": article_count,
        }

    async def _serialize_run_admin(self, session, run: NewsIngestionRun) -> dict[str, Any]:
        source_label = None
        if run.site_feed_id is not None:
            feed = await session.get(NewsSiteFeed, run.site_feed_id)
            if feed is not None:
                source_label = feed.title or feed.feed_url
        elif run.crawl_source_id is not None:
            source = await session.get(NewsCrawlSource, run.crawl_source_id)
            if source is not None:
                source_label = source.listing_url
        return {
            "id": run.id,
            "source_type": run.source_type,
            "site_feed_id": run.site_feed_id,
            "crawl_source_id": run.crawl_source_id,
            "source_label": source_label,
            "status": run.status,
            "fetched_count": run.fetched_count,
            "stored_count": run.stored_count,
            "filtered_count": run.filtered_count,
            "error": run.error,
            "started_at": utc_naive_to_local_iso(run.started_at),
            "finished_at": utc_naive_to_local_iso(run.finished_at),
        }

    async def _serialize_feed_source(self, session, feed_id: int, *, user_id: int) -> dict[str, Any]:
        feed = await session.get(NewsSiteFeed, feed_id)
        if feed is None:
            raise ValueError("News source not found")
        site = await session.get(NewsSite, feed.site_id)
        subscription = await self._get_subscription(session, user_id=user_id, site_feed_id=feed.id)
        return {
            "source_type": "rss",
            "id": feed.id,
            "site_id": site.id if site else None,
            "site_name": site.display_name if site else None,
            "site_url": site.homepage_url if site else None,
            "domain": site.domain if site else None,
            "feed_url": feed.feed_url,
            "title": feed.title,
            "kind": feed.kind,
            "discovery_method": feed.discovery_method,
            "validation_status": feed.validation_status,
            "validation_error": feed.validation_error,
            "enabled": subscription.enabled,
            "poll_interval_minutes": feed.poll_interval_minutes,
            "is_public": feed.is_public,
            "last_success_at": utc_naive_to_local_iso(feed.last_success_at),
            "last_failure_at": utc_naive_to_local_iso(feed.last_failure_at),
            "created_at": utc_naive_to_local_iso(feed.created_at),
            "updated_at": utc_naive_to_local_iso(feed.updated_at),
        }

    async def _serialize_crawl_source(self, session, source_id: int, *, user_id: int) -> dict[str, Any]:
        source = await session.get(NewsCrawlSource, source_id)
        if source is None:
            raise ValueError("News source not found")
        site = await session.get(NewsSite, source.site_id)
        subscription = await self._get_subscription(session, user_id=user_id, crawl_source_id=source.id)
        return {
            "source_type": "crawl",
            "id": source.id,
            "site_id": site.id if site else None,
            "site_name": site.display_name if site else None,
            "site_url": site.homepage_url if site else None,
            "domain": site.domain if site else None,
            "listing_url": source.listing_url,
            "article_link_selector": source.article_link_selector,
            "content_selector": source.content_selector,
            "excerpt_selector": source.excerpt_selector,
            "pagination_config": source.pagination_config,
            "validation_status": source.validation_status,
            "validation_error": source.validation_error,
            "enabled": subscription.enabled,
            "poll_interval_minutes": source.poll_interval_minutes,
            "is_public": source.is_public,
            "last_success_at": utc_naive_to_local_iso(source.last_success_at),
            "last_failure_at": utc_naive_to_local_iso(source.last_failure_at),
            "created_at": utc_naive_to_local_iso(source.created_at),
            "updated_at": utc_naive_to_local_iso(source.updated_at),
        }

    def _serialize_preference(self, preference: NewsUserPreference) -> dict[str, Any]:
        return {
            "blocked_topics_text": preference.blocked_topics_text,
            "blocked_labels": [str(item) for item in (preference.blocked_labels or [])],
            "updated_at": utc_naive_to_local_iso(preference.updated_at),
        }

    def _serialize_feed_item(
        self,
        article: NewsArticle,
        *,
        sources: list[dict[str, Any]],
        semantics: NewsArticleSemantic | None,
        include_content: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "id": article.id,
            "title": article.title,
            "excerpt": article.llm_summary or article.excerpt,
            "original_excerpt": article.excerpt,
            "llm_summary": article.llm_summary,
            "published_at": article.published_at.isoformat() if article.published_at else None,
            "canonical_url": article.canonical_url,
            "image_url": article.image_url,
            "language": article.language,
            "source_labels": [item["label"] for item in sources],
            "sources": sources,
            "topics": _display_topics(semantics),
            "tickers": [str(item) for item in (semantics.tickers if semantics else [])],
            "sectors": [str(item) for item in (semantics.sectors if semantics else [])],
            "importance": semantics.importance if semantics else None,
            "sentiment": semantics.sentiment if semantics else None,
            "event_type": _semantic_event_type(semantics),
            "event_labels": _semantic_event_labels(semantics),
            "matched_tickers": [],
            "why_relevant": [],
            "story_key": _semantic_story_key(semantics, article),
            "story_source_count": len(
                {
                    f"{item.get('source_type')}:{item.get('source_id')}"
                    for item in sources
                    if item and item.get("source_id") is not None
                }
            )
            or len({str(item.get("article_url") or "") for item in sources if item})
            or 1,
            "related_article_ids": [],
        }
        if include_content:
            payload["content_text"] = article.content_text
        return payload

    def _parse_cursor(self, cursor: str | None) -> tuple[datetime | None, int | None]:
        if not cursor:
            return None, None
        try:
            raw_dt, raw_id = cursor.split("|", 1)
            return datetime.fromisoformat(raw_dt), int(raw_id)
        except Exception:
            return None, None

    def _content_hash(self, title: str | None, excerpt: str | None, content_text: str | None) -> str | None:
        text = "\n".join(part for part in [title or "", excerpt or "", content_text or ""] if part).strip()
        if not text:
            return None
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_app_settings_yaml(self) -> dict[str, Any]:
        path = Path(settings.settings_yaml_path)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}

    def _write_app_settings_yaml(self, payload: dict[str, Any]) -> None:
        path = Path(settings.settings_yaml_path)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    def _should_normalize_legacy_feed_title(self, title: str | None, site: NewsSite | None) -> bool:
        if site is None:
            return False
        if not title or not title.strip():
            return True

        cleaned_title = title.strip()
        lowered_title = cleaned_title.casefold()
        domain = (site.domain or "").casefold()
        display_name = (site.display_name or "").casefold()

        if domain and domain in lowered_title:
            return False
        if display_name and display_name in lowered_title:
            return False
        if any(token in lowered_title for token in ("rss", "feed", "atom")):
            return False

        word_count = len(re.findall(r"\w+", cleaned_title, flags=re.UNICODE))
        return word_count >= LEGACY_FEED_TITLE_WORD_THRESHOLD or len(cleaned_title) >= LEGACY_FEED_TITLE_LENGTH_THRESHOLD

    def _should_repair_article_content(
        self,
        article: NewsArticle,
    ) -> bool:
        content_text = article.content_text
        if not content_text:
            return True
        if extract_domain(article.canonical_url) in ALWAYS_REFRESH_DETAIL_DOMAINS:
            return True
        excerpt = (article.excerpt or "").strip()
        normalized_content = re.sub(r"\s+", " ", content_text).strip().casefold()
        normalized_excerpt = re.sub(r"\s+", " ", excerpt).strip().casefold()
        if (
            normalized_excerpt
            and normalized_content
            and len(normalized_content) <= max(280, len(normalized_excerpt) + 40)
            and (
                normalized_content == normalized_excerpt
                or normalized_content in normalized_excerpt
                or normalized_excerpt in normalized_content
            )
        ):
            return True
        lowered = content_text.lower()
        if any(marker in lowered for marker in CONTENT_REPAIR_MARKERS):
            return True

        title_words = {
            token
            for token in re.findall(r"\w+", (article.title or "").lower(), flags=re.UNICODE)
            if len(token) >= 4
        }
        content_words = {
            token
            for token in re.findall(r"\w+", lowered, flags=re.UNICODE)
            if len(token) >= 4
        }
        if title_words and len(title_words & content_words) <= 1 and len(content_text) < 2000:
            return True
        return False

    async def _repair_article_content_if_needed(self, session, article: NewsArticle, *, force_refresh: bool = False) -> bool:
        if not force_refresh and not self._should_repair_article_content(article):
            return False
        try:
            html_text = await fetch_text(self._client, article.canonical_url)
            payload = extract_article_payload(article.canonical_url, html_text)
        except Exception:
            return False

        content_text = payload.get("content_text")
        if not content_text:
            return False
        if content_text == article.content_text and payload.get("excerpt") == article.excerpt:
            return force_refresh

        article.excerpt = payload.get("excerpt") or article.excerpt
        article.content_text = content_text
        article.image_url = payload.get("image_url") or article.image_url
        article.language = payload.get("language") or article.language
        article.published_at = payload.get("published_at") or article.published_at
        article.content_hash = self._content_hash(article.title, article.excerpt, article.content_text)
        await session.commit()
        await session.refresh(article)
        return True

    async def _upsert_article_semantics(self, session, article: NewsArticle) -> None:
        semantics = await classify_article(article.title, article.excerpt, article.content_text)
        raw_payload = dict(semantics["raw_payload"]) if isinstance(semantics.get("raw_payload"), dict) else {}
        raw_payload["event_type"] = semantics.get("event_type")
        raw_payload["event_labels"] = semantics.get("event_labels") or []
        raw_payload["story_key"] = _compute_story_key(
            article.title,
            [str(item) for item in semantics.get("tickers", [])],
            article.published_at,
        )
        semantic_row = (
            await session.execute(select(NewsArticleSemantic).where(NewsArticleSemantic.article_id == article.id))
        ).scalar_one_or_none()
        if semantic_row is None:
            semantic_row = NewsArticleSemantic(article_id=article.id)
            session.add(semantic_row)
        semantic_row.topics = semantics["topics"]
        semantic_row.tickers = semantics["tickers"]
        semantic_row.sectors = semantics["sectors"]
        semantic_row.importance = semantics["importance"]
        semantic_row.sentiment = semantics["sentiment"]
        semantic_row.raw_payload = raw_payload
        semantic_row.classified_at = utc_now()
        await session.flush()
