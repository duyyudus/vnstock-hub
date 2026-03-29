from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.deps import get_current_admin_user, get_current_user, get_current_user_optional
from app.services.news import news_service
from app.services.news.service import utc_naive_to_local_iso, utc_now


router = APIRouter(prefix="/news", tags=["news"])


NewsSourceKind = Literal["rss", "crawl"]
NewsDiscoveryMethod = Literal["homepage", "manual", "default_pack", "sitemap"]
NewsValidationStatus = Literal["pending", "valid", "invalid"]
NewsSortMode = Literal["latest", "relevance"]
NewsScopeMode = Literal["all", "portfolio", "bookmarks"]
NewsGroupMode = Literal["article", "story"]
NewsEventType = Literal[
    "earnings",
    "dividend",
    "capital_raise",
    "insider_trading",
    "management_change",
    "regulatory",
    "mna",
    "analyst_view",
    "macro_policy",
    "other",
]
NewsImportanceLevel = Literal["low", "medium", "high"]


class NewsSiteResponse(BaseModel):
    id: int
    domain: str
    homepage_url: str
    display_name: str | None = None
    is_public: bool
    created_at: str | None = None
    updated_at: str | None = None


class NewsSourceSummaryResponse(BaseModel):
    id: int
    kind: NewsSourceKind
    title: str | None = None
    enabled: bool
    validation_status: NewsValidationStatus
    last_validated_at: str | None = None
    last_error: str | None = None
    poll_interval_minutes: int
    site_name: str | None = None
    site_url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class NewsRssSourceResponse(NewsSourceSummaryResponse):
    kind: Literal["rss"] = "rss"
    feed_url: str
    discovery_method: NewsDiscoveryMethod


class NewsCrawlSourceResponse(NewsSourceSummaryResponse):
    kind: Literal["crawl"] = "crawl"
    listing_url: str
    article_link_selector: str
    content_selector: str
    excerpt_selector: str | None = None
    pagination_selector: str | None = None


class NewsSourceSubscriptionResponse(BaseModel):
    id: int
    user_id: int
    source_kind: NewsSourceKind
    source_id: int
    enabled: bool
    source_title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class NewsFeedItemResponse(BaseModel):
    id: int
    title: str
    excerpt: str | None = None
    original_excerpt: str | None = None
    llm_summary: str | None = None
    canonical_url: str
    published_at: str | None = None
    language: str | None = None
    image_url: str | None = None
    source_labels: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    importance: str | None = None
    sentiment: str | None = None
    event_type: NewsEventType | None = None
    event_labels: list[str] = Field(default_factory=list)
    matched_tickers: list[str] = Field(default_factory=list)
    why_relevant: list[str] = Field(default_factory=list)
    story_key: str | None = None
    story_source_count: int = 1
    related_article_ids: list[int] = Field(default_factory=list)
    source_title: str | None = None
    source_kind: NewsSourceKind | None = None
    is_filtered_for_user: bool = False


class NewsRelatedArticleResponse(BaseModel):
    id: int
    title: str
    published_at: str | None = None
    canonical_url: str
    source_title: str | None = None


class NewsArticleDetailResponse(NewsFeedItemResponse):
    content_text: str | None = None
    source_urls: list[str] = Field(default_factory=list)
    related_article_ids: list[int] = Field(default_factory=list)
    related_articles: list[NewsRelatedArticleResponse] = Field(default_factory=list)


class NewsFeedResponse(BaseModel):
    items: list[NewsFeedItemResponse] = Field(default_factory=list)
    count: int
    next_cursor: str | None = None
    is_personalized: bool


class NewsSourcesResponse(BaseModel):
    sites: list[NewsSiteResponse] = Field(default_factory=list)
    rss_sources: list[NewsRssSourceResponse] = Field(default_factory=list)
    crawl_sources: list[NewsCrawlSourceResponse] = Field(default_factory=list)
    subscriptions: list[NewsSourceSubscriptionResponse] = Field(default_factory=list)


class NewsRssDiscoveryCandidateResponse(BaseModel):
    feed_url: str
    title: str | None = None
    site_url: str | None = None
    discovery_method: NewsDiscoveryMethod
    kind: Literal["rss", "atom"]
    validation_status: NewsValidationStatus = "valid"
    category_hint: str | None = None


class NewsCrawlDiscoveryCandidateResponse(BaseModel):
    listing_url: str
    title: str | None = None
    site_url: str | None = None
    discovery_method: NewsDiscoveryMethod
    category_hint: str | None = None


class NewsRssDiscoveryResponse(BaseModel):
    homepage_url: str
    site_title: str | None = None
    candidates: list[NewsRssDiscoveryCandidateResponse] = Field(default_factory=list)
    crawl_candidates: list[NewsCrawlDiscoveryCandidateResponse] = Field(default_factory=list)


class NewsValidationResponse(BaseModel):
    valid: bool
    message: str
    sample_title: str | None = None
    sample_excerpt: str | None = None
    candidate_count: int | None = None
    suggestions: list[str] = Field(default_factory=list)


class NewsRssDiscoveryRequest(BaseModel):
    homepage_url: str


class NewsRssSourceCreateRequest(BaseModel):
    feed_url: str
    site_url: str | None = None
    homepage_url: str | None = None
    title: str | None = None
    enabled: bool = True
    poll_interval_minutes: int | None = None
    discovery_method: str | None = None


class NewsCrawlSourceCreateRequest(BaseModel):
    listing_url: str
    article_link_selector: str
    content_selector: str
    excerpt_selector: str | None = None
    pagination_selector: str | None = None
    title: str | None = None
    site_url: str | None = None
    enabled: bool = True
    poll_interval_minutes: int | None = None


class NewsSourceUpdateRequest(BaseModel):
    title: str | None = None
    enabled: bool | None = None
    poll_interval_minutes: int | None = None


class NewsUserPreferencesResponse(BaseModel):
    blocked_topics_text: str = ""
    blocked_labels: list[str] = Field(default_factory=list)
    updated_at: str | None = None


class NewsUserPreferencesUpdateRequest(BaseModel):
    blocked_topics_text: str = ""


class NewsAdminSourceStatusResponse(BaseModel):
    kind: NewsSourceKind
    id: int
    site_id: int | None = None
    site_name: str | None = None
    site_url: str | None = None
    domain: str | None = None
    title: str | None = None
    feed_url: str | None = None
    listing_url: str | None = None
    validation_status: NewsValidationStatus
    validation_error: str | None = None
    poll_interval_minutes: int
    last_polled_at: str | None = None
    next_poll_at: str | None = None
    last_success_at: str | None = None
    last_failure_at: str | None = None
    is_public: bool
    subscription_count: int = 0
    article_count: int = 0


class NewsAdminRunResponse(BaseModel):
    id: int
    source_type: NewsSourceKind
    site_feed_id: int | None = None
    crawl_source_id: int | None = None
    source_label: str | None = None
    status: str
    fetched_count: int
    stored_count: int
    filtered_count: int
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class NewsAdminStatusResponse(BaseModel):
    worker_running: bool
    site_count: int
    rss_source_count: int
    crawl_source_count: int
    article_count: int
    run_count: int
    successful_run_count: int
    failed_run_count: int
    rss_sources: list[NewsAdminSourceStatusResponse] = Field(default_factory=list)
    crawl_sources: list[NewsAdminSourceStatusResponse] = Field(default_factory=list)
    recent_runs: list[NewsAdminRunResponse] = Field(default_factory=list)


class NewsAdminTriggerRunRequest(BaseModel):
    source_type: NewsSourceKind | None = None
    source_id: int | None = None
    public_only: bool = True


class NewsAdminTriggerRunResponse(BaseModel):
    triggered: int
    message: str


class NewsMonitoringOverviewResponse(BaseModel):
    total_sources: int = 0
    enabled_sources: int = 0
    valid_sources: int = 0
    invalid_sources: int = 0
    public_sources: int = 0
    private_sources: int = 0
    articles_total: int = 0
    articles_last_24h: int = 0
    active_runs: int = 0
    queue_size: int = 0
    last_run_at: str | None = None
    last_run_status: str | None = None
    last_run_error: str | None = None
    updated_at: str | None = None


class NewsMonitoringRunsResponse(BaseModel):
    runs: list[NewsAdminRunResponse] = Field(default_factory=list)
    count: int


class NewsMonitoringActionResponse(BaseModel):
    started: bool
    message: str
    queued_count: int | None = None
    refreshed_count: int | None = None
    timestamp: str | None = None


class NewsAdminConfigResponse(BaseModel):
    default_poll_interval_minutes: int


class NewsAdminConfigUpdateRequest(BaseModel):
    default_poll_interval_minutes: int = Field(ge=5, le=1440)


def _parse_datetime_boundary(value: str | None, *, end: bool) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        parsed_date = date.fromisoformat(value)
        return datetime.combine(parsed_date, time.max if end else time.min)


def _to_discovery_method(value: str | None) -> NewsDiscoveryMethod:
    if value == "seed":
        return "default_pack"
    if value in {"homepage", "manual", "default_pack", "sitemap"}:
        return value
    return "manual"


def _serialize_feed_item(payload: dict[str, Any]) -> NewsFeedItemResponse:
    sources = payload.get("sources") or []
    source_labels = [str(item.get("label")) for item in sources if item.get("label")]
    source_kind = sources[0]["source_type"] if sources else None
    return NewsFeedItemResponse(
        id=int(payload["id"]),
        title=str(payload["title"]),
        excerpt=payload.get("excerpt"),
        original_excerpt=payload.get("original_excerpt"),
        llm_summary=payload.get("llm_summary"),
        canonical_url=str(payload["canonical_url"]),
        published_at=payload.get("published_at"),
        language=payload.get("language"),
        image_url=payload.get("image_url"),
        source_labels=source_labels,
        topics=[str(item) for item in payload.get("topics", [])],
        tickers=[str(item) for item in payload.get("tickers", [])],
        sectors=[str(item) for item in payload.get("sectors", [])],
        importance=payload.get("importance"),
        sentiment=payload.get("sentiment"),
        event_type=payload.get("event_type"),
        event_labels=[str(item) for item in payload.get("event_labels", [])],
        matched_tickers=[str(item) for item in payload.get("matched_tickers", [])],
        why_relevant=[str(item) for item in payload.get("why_relevant", [])],
        story_key=payload.get("story_key"),
        story_source_count=int(payload.get("story_source_count") or 1),
        related_article_ids=[int(item) for item in payload.get("related_article_ids", [])],
        source_title=source_labels[0] if source_labels else None,
        source_kind=source_kind,
        is_filtered_for_user=False,
    )


def _serialize_article_detail(payload: dict[str, Any]) -> NewsArticleDetailResponse:
    base = _serialize_feed_item(payload)
    sources = payload.get("sources") or []
    return NewsArticleDetailResponse(
        **base.model_dump(),
        content_text=payload.get("content_text"),
        source_urls=[str(item.get("article_url")) for item in sources if item.get("article_url")],
        related_articles=[
            NewsRelatedArticleResponse(
                id=int(item["id"]),
                title=str(item["title"]),
                published_at=item.get("published_at"),
                canonical_url=str(item["canonical_url"]),
                source_title=item.get("source_title"),
            )
            for item in payload.get("related_articles", [])
            if isinstance(item, dict)
        ],
    )


def _serialize_rss_source(item: dict[str, Any]) -> NewsRssSourceResponse:
    last_validated_at = item.get("last_success_at") or item.get("last_failure_at")
    return NewsRssSourceResponse(
        id=int(item["id"]),
        kind="rss",
        title=item.get("title"),
        enabled=bool(item.get("enabled", True)),
        validation_status="valid" if item.get("validation_status") == "valid" else "invalid" if item.get("validation_status") == "invalid" else "pending",
        last_validated_at=last_validated_at,
        last_error=item.get("validation_error"),
        poll_interval_minutes=int(item.get("poll_interval_minutes") or settings.news_default_poll_interval_minutes),
        site_name=item.get("site_name"),
        site_url=item.get("site_url"),
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
        feed_url=str(item["feed_url"]),
        discovery_method=_to_discovery_method(item.get("discovery_method")),
    )


def _serialize_crawl_source(item: dict[str, Any]) -> NewsCrawlSourceResponse:
    pagination_config = item.get("pagination_config") or {}
    last_validated_at = item.get("last_success_at") or item.get("last_failure_at")
    return NewsCrawlSourceResponse(
        id=int(item["id"]),
        kind="crawl",
        title=item.get("site_name") or item.get("title"),
        enabled=bool(item.get("enabled", True)),
        validation_status="valid" if item.get("validation_status") == "valid" else "invalid" if item.get("validation_status") == "invalid" else "pending",
        last_validated_at=last_validated_at,
        last_error=item.get("validation_error"),
        poll_interval_minutes=int(item.get("poll_interval_minutes") or settings.news_default_poll_interval_minutes),
        site_name=item.get("site_name"),
        site_url=item.get("site_url"),
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
        listing_url=str(item["listing_url"]),
        article_link_selector=str(item["article_link_selector"]),
        content_selector=str(item["content_selector"]),
        excerpt_selector=item.get("excerpt_selector"),
        pagination_selector=pagination_config.get("selector") if isinstance(pagination_config, dict) else None,
    )


def _serialize_admin_source(item: dict[str, Any]) -> NewsAdminSourceStatusResponse:
    return NewsAdminSourceStatusResponse(
        kind=item["kind"],
        id=int(item["id"]),
        site_id=item.get("site_id"),
        site_name=item.get("site_name"),
        site_url=item.get("site_url"),
        domain=item.get("domain"),
        title=item.get("title"),
        feed_url=item.get("feed_url"),
        listing_url=item.get("listing_url"),
        validation_status="valid" if item.get("validation_status") == "valid" else "invalid" if item.get("validation_status") == "invalid" else "pending",
        validation_error=item.get("validation_error"),
        poll_interval_minutes=int(item.get("poll_interval_minutes") or settings.news_default_poll_interval_minutes),
        last_polled_at=item.get("last_polled_at"),
        next_poll_at=item.get("next_poll_at"),
        last_success_at=item.get("last_success_at"),
        last_failure_at=item.get("last_failure_at"),
        is_public=bool(item.get("is_public")),
        subscription_count=int(item.get("subscription_count") or 0),
        article_count=int(item.get("article_count") or 0),
    )


def _serialize_admin_run(item: dict[str, Any]) -> NewsAdminRunResponse:
    return NewsAdminRunResponse(
        id=int(item["id"]),
        source_type=item["source_type"],
        site_feed_id=item.get("site_feed_id"),
        crawl_source_id=item.get("crawl_source_id"),
        source_label=item.get("source_label"),
        status=str(item["status"]),
        fetched_count=int(item.get("fetched_count") or 0),
        stored_count=int(item.get("stored_count") or 0),
        filtered_count=int(item.get("filtered_count") or 0),
        error=item.get("error"),
        started_at=item.get("started_at"),
        finished_at=item.get("finished_at"),
    )


@router.get("/feed", response_model=NewsFeedResponse)
async def get_news_feed(
    source: str | None = Query(default=None),
    topic: str | None = Query(default=None),
    ticker: str | None = Query(default=None),
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=50),
    sort: NewsSortMode = Query(default="latest"),
    scope: NewsScopeMode = Query(default="all"),
    bookmark_group_id: int | None = Query(default=None, ge=1),
    event_type: NewsEventType | None = Query(default=None),
    importance: NewsImportanceLevel | None = Query(default=None),
    group_by: NewsGroupMode = Query(default="article"),
    current_user=Depends(get_current_user_optional),
):
    try:
        payload = await news_service.get_feed(
            user_id=current_user.id if current_user else None,
            source=source,
            topic=topic,
            ticker=ticker,
            date_from=_parse_datetime_boundary(from_, end=False),
            date_to=_parse_datetime_boundary(to, end=True),
            cursor=cursor,
            limit=limit,
            sort=sort,
            scope=scope,
            bookmark_group_id=bookmark_group_id,
            event_type=event_type,
            importance=importance,
            group_by=group_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return NewsFeedResponse(
        items=[_serialize_feed_item(item) for item in payload["items"]],
        count=int(payload["count"]),
        next_cursor=payload.get("next_cursor"),
        is_personalized=bool(current_user),
    )


@router.get("/articles/{article_id}", response_model=NewsArticleDetailResponse)
async def get_news_article(article_id: int, current_user=Depends(get_current_user_optional)):
    payload = await news_service.get_article_detail(article_id, user_id=current_user.id if current_user else None)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="News article not found")
    return _serialize_article_detail(payload)


@router.post("/articles/{article_id}/summary", response_model=NewsFeedItemResponse)
async def generate_news_article_summary(
    article_id: int,
    force_refresh: bool = Query(default=False),
    current_user=Depends(get_current_user_optional),
):
    try:
        payload = await news_service.generate_article_summary(
            article_id,
            user_id=current_user.id if current_user else None,
            force_refresh=force_refresh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="News article not found")
    return _serialize_feed_item(payload)


@router.post("/articles/{article_id}/refresh-content", response_model=NewsArticleDetailResponse)
async def refresh_news_article_content(article_id: int, current_user=Depends(get_current_user_optional)):
    try:
        payload = await news_service.refresh_article_content(
            article_id,
            user_id=current_user.id if current_user else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="News article not found")
    return _serialize_article_detail(payload)


@router.post("/rss/discover", response_model=NewsRssDiscoveryResponse)
async def discover_news_rss(payload: NewsRssDiscoveryRequest, current_user=Depends(get_current_user)):
    del current_user
    candidates = await news_service.discover_rss_feeds(payload.homepage_url)
    crawl_candidates = await news_service.discover_crawl_listings(payload.homepage_url) if not candidates else []
    return NewsRssDiscoveryResponse(
        homepage_url=payload.homepage_url,
        site_title=None,
        candidates=[
            NewsRssDiscoveryCandidateResponse(
                feed_url=str(item["feed_url"]),
                title=item.get("title"),
                site_url=item.get("site_url"),
                discovery_method=_to_discovery_method(item.get("discovery_method")),
                kind="atom" if item.get("kind") == "atom" else "rss",
                validation_status="valid",
                category_hint=item.get("candidate_title"),
            )
            for item in candidates
        ],
        crawl_candidates=[
            NewsCrawlDiscoveryCandidateResponse(
                listing_url=str(item["listing_url"]),
                title=item.get("title"),
                site_url=item.get("site_url"),
                discovery_method=_to_discovery_method(item.get("discovery_method")),
                category_hint=item.get("title"),
            )
            for item in crawl_candidates
        ],
    )


@router.post("/rss/validate", response_model=NewsValidationResponse)
async def validate_news_rss(payload: NewsRssSourceCreateRequest, current_user=Depends(get_current_user)):
    del current_user
    result = await news_service.validate_rss_feed(payload.feed_url, site_url=payload.site_url or payload.homepage_url)
    sample_entries = result.get("sample_entries") or []
    return NewsValidationResponse(
        valid=True,
        message="Feed validated successfully.",
        sample_title=sample_entries[0]["title"] if sample_entries else result.get("title"),
        sample_excerpt=None,
        candidate_count=int(result.get("entry_count") or len(sample_entries)),
        suggestions=[],
    )


@router.post("/crawl/validate", response_model=NewsValidationResponse)
async def validate_news_crawl(payload: NewsCrawlSourceCreateRequest, current_user=Depends(get_current_user)):
    del current_user
    result = await news_service.validate_crawl_source(
        listing_url=payload.listing_url,
        article_link_selector=payload.article_link_selector,
        content_selector=payload.content_selector,
        excerpt_selector=payload.excerpt_selector,
    )
    sample_articles = result.get("sample_articles") or []
    return NewsValidationResponse(
        valid=True,
        message="Crawl source validated successfully.",
        sample_title=sample_articles[0]["title"] if sample_articles else None,
        sample_excerpt=sample_articles[0]["excerpt"] if sample_articles else None,
        candidate_count=int(result.get("matched_article_count") or len(sample_articles)),
        suggestions=[str(item) for item in result.get("heuristic_selectors", [])],
    )


@router.get("/sources", response_model=NewsSourcesResponse)
async def get_news_sources(current_user=Depends(get_current_user)):
    items = await news_service.list_user_sources(current_user.id)
    rss_sources = [_serialize_rss_source(item) for item in items if item.get("source_type") == "rss"]
    crawl_sources = [_serialize_crawl_source(item) for item in items if item.get("source_type") == "crawl"]
    subscriptions = [
        NewsSourceSubscriptionResponse(
            id=item.id,
            user_id=current_user.id,
            source_kind=item.kind,
            source_id=item.id,
            enabled=item.enabled,
            source_title=item.title or item.site_name,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
        for item in [*rss_sources, *crawl_sources]
    ]
    return NewsSourcesResponse(
        sites=[],
        rss_sources=rss_sources,
        crawl_sources=crawl_sources,
        subscriptions=subscriptions,
    )


@router.post("/sources/rss", response_model=NewsRssSourceResponse)
async def create_news_rss_source(payload: NewsRssSourceCreateRequest, current_user=Depends(get_current_user)):
    item = await news_service.create_rss_source(
        user_id=current_user.id,
        feed_url=payload.feed_url,
        site_url=payload.site_url or payload.homepage_url,
        poll_interval_minutes=payload.poll_interval_minutes,
    )
    if payload.enabled is False:
        item = await news_service.update_source(
            user_id=current_user.id,
            source_type="rss",
            source_id=int(item["id"]),
            enabled=False,
            title=payload.title,
        )
    elif payload.title:
        item = await news_service.update_source(
            user_id=current_user.id,
            source_type="rss",
            source_id=int(item["id"]),
            title=payload.title,
        )
    return _serialize_rss_source(item)


@router.post("/sources/crawl", response_model=NewsCrawlSourceResponse)
async def create_news_crawl_source(payload: NewsCrawlSourceCreateRequest, current_user=Depends(get_current_user)):
    item = await news_service.create_crawl_source(
        user_id=current_user.id,
        listing_url=payload.listing_url,
        article_link_selector=payload.article_link_selector,
        content_selector=payload.content_selector,
        excerpt_selector=payload.excerpt_selector,
        pagination_config={"selector": payload.pagination_selector} if payload.pagination_selector else None,
        poll_interval_minutes=payload.poll_interval_minutes,
    )
    if payload.enabled is False or payload.title:
        item = await news_service.update_source(
            user_id=current_user.id,
            source_type="crawl",
            source_id=int(item["id"]),
            enabled=payload.enabled,
            title=payload.title,
        )
    return _serialize_crawl_source(item)


@router.patch("/sources/{source_type}/{source_id}", response_model=NewsSourceSummaryResponse)
async def update_news_source(
    source_type: NewsSourceKind,
    source_id: int,
    payload: NewsSourceUpdateRequest,
    current_user=Depends(get_current_user),
):
    item = await news_service.update_source(
        user_id=current_user.id,
        source_type=source_type,
        source_id=source_id,
        enabled=payload.enabled,
        title=payload.title,
        poll_interval_minutes=payload.poll_interval_minutes,
    )
    if source_type == "rss":
        return _serialize_rss_source(item)
    return _serialize_crawl_source(item)


@router.delete("/sources/{source_type}/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_news_source(
    source_type: NewsSourceKind,
    source_id: int,
    current_user=Depends(get_current_user),
):
    await news_service.delete_source(user_id=current_user.id, source_type=source_type, source_id=source_id)


@router.get("/preferences", response_model=NewsUserPreferencesResponse)
async def get_news_preferences(current_user=Depends(get_current_user)):
    payload = await news_service.get_user_preferences(current_user.id)
    return NewsUserPreferencesResponse(
        blocked_topics_text=str(payload.get("blocked_topics_text") or ""),
        blocked_labels=[str(item) for item in payload.get("blocked_labels", [])],
        updated_at=payload.get("updated_at"),
    )


@router.patch("/preferences", response_model=NewsUserPreferencesResponse)
async def update_news_preferences(
    payload: NewsUserPreferencesUpdateRequest,
    current_user=Depends(get_current_user),
):
    result = await news_service.update_user_preferences(current_user.id, payload.blocked_topics_text)
    return NewsUserPreferencesResponse(
        blocked_topics_text=str(result.get("blocked_topics_text") or ""),
        blocked_labels=[str(item) for item in result.get("blocked_labels", [])],
        updated_at=result.get("updated_at"),
    )


@router.get("/admin/status", response_model=NewsAdminStatusResponse)
async def get_news_admin_status(current_user=Depends(get_current_admin_user)):
    del current_user
    payload = await news_service.get_admin_status()
    return NewsAdminStatusResponse(
        worker_running=bool(payload["worker_running"]),
        site_count=int(payload["site_count"]),
        rss_source_count=int(payload["rss_source_count"]),
        crawl_source_count=int(payload["crawl_source_count"]),
        article_count=int(payload["article_count"]),
        run_count=int(payload["run_count"]),
        successful_run_count=int(payload["successful_run_count"]),
        failed_run_count=int(payload["failed_run_count"]),
        rss_sources=[_serialize_admin_source(item) for item in payload["rss_sources"]],
        crawl_sources=[_serialize_admin_source(item) for item in payload["crawl_sources"]],
        recent_runs=[_serialize_admin_run(item) for item in payload["recent_runs"]],
    )


@router.post("/admin/run", response_model=NewsAdminTriggerRunResponse)
async def trigger_news_admin_run(
    payload: NewsAdminTriggerRunRequest,
    current_user=Depends(get_current_admin_user),
):
    del current_user
    result = await news_service.trigger_admin_run(
        source_type=payload.source_type,
        source_id=payload.source_id,
        public_only=payload.public_only,
    )
    return NewsAdminTriggerRunResponse(
        triggered=int(result["triggered"]),
        message=str(result["message"]),
    )


@router.get("/admin/overview", response_model=NewsMonitoringOverviewResponse)
async def get_news_monitoring_overview(current_user=Depends(get_current_admin_user)):
    del current_user
    payload = await news_service.get_admin_status()
    all_sources = [*payload["rss_sources"], *payload["crawl_sources"]]
    valid_sources = sum(1 for item in all_sources if item.get("validation_status") == "valid")
    invalid_sources = sum(1 for item in all_sources if item.get("validation_status") == "invalid")
    public_sources = sum(1 for item in all_sources if item.get("is_public"))
    private_sources = len(all_sources) - public_sources
    recent_runs = payload["recent_runs"]
    active_runs = int(payload.get("active_run_count") or 0)
    last_run = recent_runs[0] if recent_runs else None
    return NewsMonitoringOverviewResponse(
        total_sources=len(all_sources),
        enabled_sources=len(all_sources),
        valid_sources=valid_sources,
        invalid_sources=invalid_sources,
        public_sources=public_sources,
        private_sources=private_sources,
        articles_total=int(payload["article_count"]),
        articles_last_24h=0,
        active_runs=active_runs,
        queue_size=0,
        last_run_at=last_run.get("started_at") if last_run else None,
        last_run_status=last_run.get("status") if last_run else None,
        last_run_error=last_run.get("error") if last_run else None,
        updated_at=utc_naive_to_local_iso(utc_now()),
    )


@router.get("/admin/config", response_model=NewsAdminConfigResponse)
async def get_news_admin_config(current_user=Depends(get_current_admin_user)):
    del current_user
    payload = news_service.get_admin_config()
    return NewsAdminConfigResponse(
        default_poll_interval_minutes=int(payload["default_poll_interval_minutes"]),
    )


@router.patch("/admin/config", response_model=NewsAdminConfigResponse)
async def update_news_admin_config(
    payload: NewsAdminConfigUpdateRequest,
    current_user=Depends(get_current_admin_user),
):
    del current_user
    result = news_service.update_admin_config(
        default_poll_interval_minutes=payload.default_poll_interval_minutes,
    )
    return NewsAdminConfigResponse(
        default_poll_interval_minutes=int(result["default_poll_interval_minutes"]),
    )


@router.get("/admin/sources", response_model=NewsSourcesResponse)
async def get_news_monitoring_sources(current_user=Depends(get_current_admin_user)):
    del current_user
    payload = await news_service.get_admin_status()
    site_map: dict[int, NewsSiteResponse] = {}
    for item in [*payload["rss_sources"], *payload["crawl_sources"]]:
        site_id = item.get("site_id")
        if site_id is None or site_id in site_map:
            continue
        site_map[int(site_id)] = NewsSiteResponse(
            id=int(site_id),
            domain=str(item.get("domain") or ""),
            homepage_url=str(item.get("site_url") or ""),
            display_name=item.get("site_name"),
            is_public=bool(item.get("is_public")),
            created_at=None,
            updated_at=None,
        )
    return NewsSourcesResponse(
        sites=list(site_map.values()),
        rss_sources=[
            _serialize_rss_source(
                {
                    **item,
                    "source_type": "rss",
                    "created_at": None,
                    "updated_at": None,
                }
            )
            for item in payload["rss_sources"]
        ],
        crawl_sources=[
            _serialize_crawl_source(
                {
                    **item,
                    "source_type": "crawl",
                    "created_at": None,
                    "updated_at": None,
                }
            )
            for item in payload["crawl_sources"]
        ],
        subscriptions=[],
    )


@router.get("/admin/runs", response_model=NewsMonitoringRunsResponse)
async def get_news_monitoring_runs(
    limit: int = Query(default=12, ge=1, le=100),
    current_user=Depends(get_current_admin_user),
):
    del current_user
    payload = await news_service.get_admin_status()
    runs = [_serialize_admin_run(item) for item in payload["recent_runs"][:limit]]
    return NewsMonitoringRunsResponse(runs=runs, count=len(runs))


@router.post("/admin/ingest", response_model=NewsMonitoringActionResponse)
async def trigger_news_monitoring_ingest(current_user=Depends(get_current_admin_user)):
    del current_user
    result = await news_service.trigger_admin_run(public_only=True)
    return NewsMonitoringActionResponse(
        started=True,
        message=str(result["message"]),
        queued_count=int(result["triggered"]),
        timestamp=utc_naive_to_local_iso(utc_now()),
    )


@router.post("/admin/refresh", response_model=NewsMonitoringActionResponse)
async def refresh_news_monitoring_sources(current_user=Depends(get_current_admin_user)):
    del current_user
    await news_service.ensure_public_sources()
    payload = await news_service.get_admin_status()
    refreshed_count = int(payload["rss_source_count"]) + int(payload["crawl_source_count"])
    return NewsMonitoringActionResponse(
        started=True,
        message="Public news source cache refreshed.",
        refreshed_count=refreshed_count,
        timestamp=utc_naive_to_local_iso(utc_now()),
    )


@router.post("/admin/repair-rss-titles", response_model=NewsMonitoringActionResponse)
async def repair_news_monitoring_rss_titles(current_user=Depends(get_current_admin_user)):
    del current_user
    repaired_count = await news_service.normalize_legacy_rss_titles()
    return NewsMonitoringActionResponse(
        started=True,
        message=(
            f"Normalized {repaired_count} legacy RSS source title{'s' if repaired_count != 1 else ''}."
            if repaired_count
            else "No legacy RSS source titles needed normalization."
        ),
        refreshed_count=repaired_count,
        timestamp=utc_naive_to_local_iso(utc_now()),
    )


@router.post("/admin/apply-default-poll-interval", response_model=NewsMonitoringActionResponse)
async def apply_news_monitoring_default_poll_interval(current_user=Depends(get_current_admin_user)):
    del current_user
    applied_count = await news_service.apply_default_poll_interval_to_existing_sources()
    return NewsMonitoringActionResponse(
        started=True,
        message=(
            f"Applied the default poll interval to {applied_count} existing source{'s' if applied_count != 1 else ''}."
            if applied_count
            else "All existing sources already use the current default poll interval."
        ),
        refreshed_count=applied_count,
        timestamp=utc_naive_to_local_iso(utc_now()),
    )
