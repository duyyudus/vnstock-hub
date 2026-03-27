from datetime import datetime

import pytest
from httpx import AsyncClient

from app.db.models import (
    NewsArticle,
    NewsArticleSemantic,
    NewsArticleSource,
    NewsSite,
    NewsSiteFeed,
    NewsUserPreference,
)
from app.services.news import news_service


async def _register_and_auth(client: AsyncClient, email: str) -> tuple[dict[str, str], int]:
    password = "password123"
    response = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert response.status_code == 201
    payload = response.json()
    token = payload["access_token"]
    return {"Authorization": f"Bearer {token}"}, int(payload["user"]["id"])


@pytest.mark.asyncio
async def test_public_news_feed_returns_public_articles(client, db_session):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example News",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Example RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://example.com/articles/one",
        title="ABC posts strong earnings",
        excerpt="Quarterly earnings improved.",
        content_text="ABC earnings and revenue growth.",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        language="en",
        content_hash="hash-1",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    db_session.add(
        NewsArticleSemantic(
            article_id=article.id,
            topics=["earnings"],
            tickers=["ABC"],
            sectors=["technology"],
            importance="high",
            sentiment="positive",
            raw_payload={},
        )
    )
    await db_session.commit()

    response = await client.get("/api/v1/news/feed")
    assert response.status_code == 200
    payload = response.json()
    assert payload["is_personalized"] is False
    assert payload["count"] == 1
    assert payload["items"][0]["title"] == "ABC posts strong earnings"
    assert payload["items"][0]["source_labels"] == ["example.com"]
    assert payload["items"][0]["tickers"] == ["ABC"]


@pytest.mark.asyncio
async def test_personalized_feed_excludes_blocked_topics(client, db_session):
    headers, user_id = await _register_and_auth(client, "news-blocked@example.com")

    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example News",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Example RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://example.com/articles/two",
        title="Bank dividend announced",
        excerpt="Dividend and record date update.",
        content_text="Dividend details for ABC bank shareholders.",
        published_at=datetime(2026, 3, 26, 11, 0, 0),
        language="en",
        content_hash="hash-2",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    db_session.add(
        NewsArticleSemantic(
            article_id=article.id,
            topics=["dividend", "banking"],
            tickers=["ABC"],
            sectors=["banking"],
            importance="high",
            sentiment="neutral",
            raw_payload={},
        )
    )
    db_session.add(NewsUserPreference(user_id=user_id, blocked_topics_text="dividend", blocked_labels=["dividend"]))
    await db_session.commit()

    response = await client.get("/api/v1/news/feed", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["is_personalized"] is True
    assert payload["items"] == []


@pytest.mark.asyncio
async def test_news_feed_source_filter_matches_site_domain(client, db_session):
    cafef_site = NewsSite(
        domain="cafef.vn",
        homepage_url="https://cafef.vn",
        display_name="CafeF",
        is_public=True,
    )
    vne_site = NewsSite(
        domain="vneconomy.vn",
        homepage_url="https://vneconomy.vn",
        display_name="VnEconomy",
        is_public=True,
    )
    db_session.add_all([cafef_site, vne_site])
    await db_session.flush()

    cafef_feed = NewsSiteFeed(
        site_id=cafef_site.id,
        feed_url="https://cafef.vn/rss.chn",
        title="CafeF RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    vne_feed = NewsSiteFeed(
        site_id=vne_site.id,
        feed_url="https://vneconomy.vn/tai-chinh.rss",
        title="VnEconomy RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add_all([cafef_feed, vne_feed])
    await db_session.flush()

    cafef_article = NewsArticle(
        canonical_url="https://cafef.vn/article-a.chn",
        title="CafeF article",
        excerpt="CafeF excerpt",
        content_text="CafeF content.",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        language="vi",
        content_hash="cafef-hash",
    )
    vne_article = NewsArticle(
        canonical_url="https://vneconomy.vn/article-b.htm",
        title="VnEconomy article",
        excerpt="VnEconomy excerpt",
        content_text="VnEconomy content.",
        published_at=datetime(2026, 3, 26, 9, 0, 0),
        language="vi",
        content_hash="vne-hash",
    )
    db_session.add_all([cafef_article, vne_article])
    await db_session.flush()
    db_session.add_all(
        [
            NewsArticleSource(article_id=cafef_article.id, site_feed_id=cafef_feed.id, article_url=cafef_article.canonical_url),
            NewsArticleSource(article_id=vne_article.id, site_feed_id=vne_feed.id, article_url=vne_article.canonical_url),
        ]
    )
    await db_session.commit()

    response = await client.get("/api/v1/news/feed?source=cafef.vn")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["items"][0]["canonical_url"] == "https://cafef.vn/article-a.chn"


@pytest.mark.asyncio
async def test_news_sources_are_private_per_user(client, monkeypatch):
    user_one_headers, _ = await _register_and_auth(client, "news-user-one@example.com")
    user_two_headers, _ = await _register_and_auth(client, "news-user-two@example.com")

    async def _fake_validate_feed(feed_url: str, site_url: str | None = None):
        return {
            "feed_url": feed_url,
            "site_url": site_url,
            "kind": "rss",
            "title": "Private Feed",
            "entry_count": 3,
            "sample_entries": [{"title": "Sample", "link": "https://example.com/a", "published_at": None}],
        }

    monkeypatch.setattr(news_service, "validate_rss_feed", _fake_validate_feed)

    create_response = await client.post(
        "/api/v1/news/sources/rss",
        json={
            "feed_url": "https://private.example.com/rss.xml",
            "site_url": "https://private.example.com",
            "poll_interval_minutes": 45,
        },
        headers=user_one_headers,
    )
    assert create_response.status_code == 200
    assert create_response.json()["feed_url"] == "https://private.example.com/rss.xml"

    list_one = await client.get("/api/v1/news/sources", headers=user_one_headers)
    assert list_one.status_code == 200
    assert len(list_one.json()["rss_sources"]) == 1

    list_two = await client.get("/api/v1/news/sources", headers=user_two_headers)
    assert list_two.status_code == 200
    assert list_two.json()["rss_sources"] == []


@pytest.mark.asyncio
async def test_news_rss_discovery_returns_multiple_candidates(client, monkeypatch):
    headers, _ = await _register_and_auth(client, "news-discovery@example.com")

    async def _fake_discover(homepage_url: str):
        return [
            {
                "feed_url": "https://example.com/rss.xml",
                "title": "Main Feed",
                "site_url": homepage_url,
                "discovery_method": "homepage",
                "kind": "rss",
                "candidate_title": "Main",
            },
            {
                "feed_url": "https://example.com/markets.xml",
                "title": "Markets Feed",
                "site_url": homepage_url,
                "discovery_method": "homepage",
                "kind": "atom",
                "candidate_title": "Markets",
            },
        ]

    monkeypatch.setattr(news_service, "discover_rss_feeds", _fake_discover)

    response = await client.post(
        "/api/v1/news/rss/discover",
        json={"homepage_url": "https://example.com"},
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["homepage_url"] == "https://example.com"
    assert len(payload["candidates"]) == 2
    assert payload["candidates"][1]["kind"] == "atom"
    assert payload["crawl_candidates"] == []


@pytest.mark.asyncio
async def test_news_rss_discovery_returns_crawl_candidates_when_no_feeds_exist(client, monkeypatch):
    headers, _ = await _register_and_auth(client, "news-discovery-crawl@example.com")

    async def _fake_discover(homepage_url: str):
        return []

    async def _fake_discover_crawl(homepage_url: str):
        return [
            {
                "listing_url": "https://example.com/kinh-te/",
                "title": "Kinh Te",
                "site_url": homepage_url,
                "discovery_method": "sitemap",
            },
        ]

    monkeypatch.setattr(news_service, "discover_rss_feeds", _fake_discover)
    monkeypatch.setattr(news_service, "discover_crawl_listings", _fake_discover_crawl)

    response = await client.post(
        "/api/v1/news/rss/discover",
        json={"homepage_url": "https://example.com"},
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["candidates"] == []
    assert payload["crawl_candidates"] == [
        {
            "listing_url": "https://example.com/kinh-te/",
            "title": "Kinh Te",
            "site_url": "https://example.com",
            "discovery_method": "sitemap",
            "category_hint": "Kinh Te",
        }
    ]


@pytest.mark.asyncio
async def test_news_crawl_validation_returns_suggestions(client, monkeypatch):
    headers, _ = await _register_and_auth(client, "news-crawl@example.com")

    async def _fake_validate_crawl(**kwargs):
        return {
            "listing_url": kwargs["listing_url"],
            "matched_article_count": 4,
            "sample_articles": [{"title": "Parsed article", "excerpt": "Short excerpt"}],
            "heuristic_selectors": ["article a", ".news-item a"],
        }

    monkeypatch.setattr(news_service, "validate_crawl_source", _fake_validate_crawl)

    response = await client.post(
        "/api/v1/news/crawl/validate",
        json={
            "listing_url": "https://example.com/news",
            "article_link_selector": "article a",
            "content_selector": "article",
        },
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True
    assert payload["candidate_count"] == 4
    assert payload["suggestions"] == ["article a", ".news-item a"]


@pytest.mark.asyncio
async def test_generate_news_article_summary_returns_persisted_llm_summary(client, db_session, monkeypatch):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example News",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Example RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://example.com/articles/summary",
        title="ABC posts strong earnings",
        excerpt="Quarterly earnings improved.",
        content_text="ABC earnings and revenue growth.",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        language="en",
        content_hash="hash-summary",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    db_session.add(
        NewsArticleSemantic(
            article_id=article.id,
            topics=["earnings"],
            tickers=["ABC"],
            sectors=["technology"],
            importance="high",
            sentiment="positive",
            raw_payload={},
        )
    )
    await db_session.commit()

    async def _fake_summarize_article(title: str, excerpt: str | None, content_text: str | None, *, language: str | None = None):
        assert language == "en"
        return "Generated card summary"

    monkeypatch.setattr("app.services.news.service.summarize_article", _fake_summarize_article)

    response = await client.post(f"/api/v1/news/articles/{article.id}/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["excerpt"] == "Generated card summary"
    assert payload["original_excerpt"] == "Quarterly earnings improved."
    assert payload["llm_summary"] == "Generated card summary"


@pytest.mark.asyncio
async def test_generate_news_article_summary_force_refresh_overwrites_existing_summary(client, db_session, monkeypatch):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example News",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Example RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://example.com/articles/summary-refresh",
        title="ABC posts strong earnings",
        excerpt="Quarterly earnings improved.",
        llm_summary="Old summary",
        content_text="ABC earnings and revenue growth.",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        language="en",
        content_hash="hash-summary-refresh",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    await db_session.commit()

    async def _fake_summarize_article(title: str, excerpt: str | None, content_text: str | None, *, language: str | None = None):
        return "Refreshed summary"

    monkeypatch.setattr("app.services.news.service.summarize_article", _fake_summarize_article)

    response = await client.post(f"/api/v1/news/articles/{article.id}/summary?force_refresh=true")
    assert response.status_code == 200
    payload = response.json()
    assert payload["excerpt"] == "Refreshed summary"
    assert payload["llm_summary"] == "Refreshed summary"


@pytest.mark.asyncio
async def test_refresh_news_article_content_returns_updated_detail(client, db_session, monkeypatch):
    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example News",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Example RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://example.com/articles/refresh",
        title="ABC posts strong earnings",
        excerpt="Old excerpt",
        content_text="Old content",
        published_at=datetime(2026, 3, 26, 10, 0, 0),
        language="en",
        content_hash="hash-refresh",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    await db_session.commit()

    async def _fake_refresh_article_content(article_id: int, *, user_id: int | None):
        assert article_id == article.id
        assert user_id is None
        return {
            "id": article.id,
            "title": article.title,
            "excerpt": "Updated excerpt",
            "original_excerpt": "Updated excerpt",
            "llm_summary": None,
            "canonical_url": article.canonical_url,
            "published_at": article.published_at.isoformat(),
            "language": article.language,
            "image_url": None,
            "source_labels": ["Example RSS"],
            "sources": [{"source_type": "rss", "source_id": feed.id, "label": "Example RSS", "article_url": article.canonical_url, "is_public": True}],
            "topics": [],
            "tickers": [],
            "sectors": [],
            "importance": None,
            "sentiment": None,
            "content_text": "Updated content",
        }

    monkeypatch.setattr(news_service, "refresh_article_content", _fake_refresh_article_content)

    response = await client.post(f"/api/v1/news/articles/{article.id}/refresh-content")
    assert response.status_code == 200
    payload = response.json()
    assert payload["excerpt"] == "Updated excerpt"
    assert payload["content_text"] == "Updated content"
    assert payload["source_urls"] == [article.canonical_url]


@pytest.mark.asyncio
async def test_news_admin_status_and_manual_run(client, monkeypatch):
    headers, _ = await _register_and_auth(client, "admin@example.com")

    async def _fake_admin_status():
        return {
            "worker_running": True,
            "site_count": 1,
            "rss_source_count": 1,
            "crawl_source_count": 0,
            "article_count": 12,
            "run_count": 2,
            "active_run_count": 0,
            "successful_run_count": 1,
            "failed_run_count": 1,
            "rss_sources": [
                {
                    "kind": "rss",
                    "id": 1,
                    "site_id": 1,
                    "site_name": "CafeF",
                    "site_url": "https://cafef.vn",
                    "domain": "cafef.vn",
                    "title": "CafeF RSS",
                    "feed_url": "https://cafef.vn/rss.chn",
                    "validation_status": "valid",
                    "validation_error": None,
                    "poll_interval_minutes": 30,
                    "last_polled_at": None,
                    "next_poll_at": None,
                    "last_success_at": None,
                    "last_failure_at": None,
                    "is_public": True,
                    "subscription_count": 0,
                    "article_count": 12,
                }
            ],
            "crawl_sources": [],
            "recent_runs": [
                {
                    "id": 2,
                    "source_type": "rss",
                    "site_feed_id": 1,
                    "crawl_source_id": None,
                    "source_label": "CafeF RSS",
                    "status": "failed",
                    "fetched_count": 0,
                    "stored_count": 0,
                    "filtered_count": 0,
                    "error": "parse error",
                    "started_at": None,
                    "finished_at": None,
                }
            ],
        }

    async def _fake_trigger_admin_run(**kwargs):
        return {"triggered": 1, "message": "Triggered ingestion for 1 public sources."}

    monkeypatch.setattr(news_service, "get_admin_status", _fake_admin_status)
    monkeypatch.setattr(news_service, "trigger_admin_run", _fake_trigger_admin_run)
    monkeypatch.setattr("app.api.v1.news.utc_now", lambda: datetime(2026, 3, 26, 12, 14, 22))

    status_response = await client.get("/api/v1/news/admin/status", headers=headers)
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["worker_running"] is True
    assert status_payload["article_count"] == 12
    assert status_payload["rss_sources"][0]["feed_url"] == "https://cafef.vn/rss.chn"

    trigger_response = await client.post("/api/v1/news/admin/run", json={"public_only": True}, headers=headers)
    assert trigger_response.status_code == 200
    assert trigger_response.json() == {
        "triggered": 1,
        "message": "Triggered ingestion for 1 public sources.",
    }

    overview_response = await client.get("/api/v1/news/admin/overview", headers=headers)
    assert overview_response.status_code == 200
    assert overview_response.json()["updated_at"] == "2026-03-26T19:14:22+07:00"
    assert overview_response.json()["active_runs"] == 0

    refresh_response = await client.post("/api/v1/news/admin/refresh", headers=headers)
    assert refresh_response.status_code == 200
    assert refresh_response.json()["timestamp"] == "2026-03-26T19:14:22+07:00"


@pytest.mark.asyncio
async def test_news_admin_repair_rss_titles(client, monkeypatch):
    headers, _ = await _register_and_auth(client, "admin@example.com")

    async def _fake_repair_titles():
        return 2

    monkeypatch.setattr(news_service, "normalize_legacy_rss_titles", _fake_repair_titles)
    monkeypatch.setattr("app.api.v1.news.utc_now", lambda: datetime(2026, 3, 26, 12, 14, 22))

    response = await client.post("/api/v1/news/admin/repair-rss-titles", headers=headers)
    assert response.status_code == 200
    assert response.json() == {
        "started": True,
        "message": "Normalized 2 legacy RSS source titles.",
        "queued_count": None,
        "refreshed_count": 2,
        "timestamp": "2026-03-26T19:14:22+07:00",
    }


@pytest.mark.asyncio
async def test_news_admin_apply_default_poll_interval(client, monkeypatch):
    headers, _ = await _register_and_auth(client, "admin@example.com")

    async def _fake_apply_default_poll_interval():
        return 3

    monkeypatch.setattr(news_service, "apply_default_poll_interval_to_existing_sources", _fake_apply_default_poll_interval)
    monkeypatch.setattr("app.api.v1.news.utc_now", lambda: datetime(2026, 3, 26, 12, 14, 22))

    response = await client.post("/api/v1/news/admin/apply-default-poll-interval", headers=headers)

    assert response.status_code == 200
    assert response.json() == {
        "started": True,
        "message": "Applied the default poll interval to 3 existing sources.",
        "queued_count": None,
        "refreshed_count": 3,
        "timestamp": "2026-03-26T19:14:22+07:00",
    }


@pytest.mark.asyncio
async def test_news_admin_config_get_returns_current_default_poll_interval(client, monkeypatch):
    headers, _ = await _register_and_auth(client, "admin@example.com")

    monkeypatch.setattr(
        news_service,
        "get_admin_config",
        lambda: {
            "default_poll_interval_minutes": 45,
        },
    )

    response = await client.get("/api/v1/news/admin/config", headers=headers)

    assert response.status_code == 200
    assert response.json() == {
        "default_poll_interval_minutes": 45,
    }


@pytest.mark.asyncio
async def test_news_admin_config_patch_updates_default_poll_interval(client, monkeypatch):
    headers, _ = await _register_and_auth(client, "admin@example.com")

    captured: dict[str, int] = {}

    def _fake_update_admin_config(*, default_poll_interval_minutes: int | None = None):
        captured["default_poll_interval_minutes"] = int(default_poll_interval_minutes or 0)
        return {
            "default_poll_interval_minutes": int(default_poll_interval_minutes or 0),
        }

    monkeypatch.setattr(news_service, "update_admin_config", _fake_update_admin_config)

    response = await client.patch(
        "/api/v1/news/admin/config",
        json={"default_poll_interval_minutes": 45},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "default_poll_interval_minutes": 45,
    }
    assert captured["default_poll_interval_minutes"] == 45
