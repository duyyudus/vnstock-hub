from datetime import datetime
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from app.db.models import NewsArticle, NewsArticleSemantic, NewsArticleSource, NewsCrawlSource, NewsIngestionRun, NewsSite, NewsSiteFeed
from app.services.news import news_service
from app.services.news import service as news_service_module
from app.services.news.service import NEWS_STALE_RUN_MINUTES, utc_naive_to_local_iso
from app.services.news.discovery import (
    discover_crawl_listings,
    discover_rss_feeds,
    extract_article_payload,
    parse_homepage_feed_candidates,
    parse_feed_entries,
    parse_feed_title,
    parse_sitemap_listing_candidates,
)


def test_parse_homepage_feed_candidates_discovers_multiple_feeds():
    html = """
    <html>
      <head>
        <link rel="alternate" type="application/rss+xml" title="Main" href="/rss.xml" />
        <link rel="alternate" type="application/atom+xml" title="Markets" href="https://example.com/markets.xml" />
      </head>
      <body>
        <a href="/feed/economy.xml">Economy feed</a>
      </body>
    </html>
    """

    candidates = parse_homepage_feed_candidates("https://example.com", html)
    urls = {candidate["feed_url"] for candidate in candidates}
    assert "https://example.com/rss.xml" in urls
    assert "https://example.com/markets.xml" in urls
    assert "https://example.com/feed/economy.xml" in urls


def test_parse_homepage_feed_candidates_detects_feed_hub_links_by_label():
    html = """
    <html>
      <body>
        <a href="/rss.html" aria-label="RSS feed directory">
          <img alt="RSS" src="/rss-icon.svg" />
        </a>
      </body>
    </html>
    """

    candidates = parse_homepage_feed_candidates("https://example.com", html)
    urls = {candidate["feed_url"] for candidate in candidates}

    assert "https://example.com/rss.html" in urls


def test_parse_feed_entries_falls_back_for_malformed_xml():
    malformed_feed = """
    <rss><channel>
      <item>
        <title>CafeF headline</title>
        <link>https://cafef.vn/story.chn</link>
        <description><![CDATA[Messy <b>markup</b> &nbsp; text]]></description>
        <pubDate>Wed, 26 Mar 2026 10:30:00 +0700</pubDate>
      </item>
    </channel></rss>
    """

    entries = parse_feed_entries(malformed_feed)
    assert len(entries) == 1
    assert entries[0].title == "CafeF headline"
    assert entries[0].link == "https://cafef.vn/story.chn"


def test_parse_feed_title_prefers_channel_title_over_item_title():
    feed_xml = """
    <rss><channel>
      <title>CafeF - Thi truong chung khoan</title>
      <item>
        <title>CafeF headline</title>
        <link>https://cafef.vn/story.chn</link>
      </item>
    </channel></rss>
    """

    assert parse_feed_title(feed_xml) == "CafeF - Thi truong chung khoan"


def test_parse_sitemap_listing_candidates_returns_stable_section_pages():
    categories_sitemap = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://example.com/kinh-te/</loc>
        <lastmod>2026-03-27T10:20:05+07:00</lastmod>
      </url>
      <url>
        <loc>https://example.com/chung-khoan/</loc>
        <lastmod>2026-03-27T10:20:05+07:00</lastmod>
      </url>
      <url>
        <loc>https://example.com/epaper/</loc>
        <lastmod>2026-03-27T10:20:05+07:00</lastmod>
      </url>
    </urlset>
    """

    candidates = parse_sitemap_listing_candidates("https://example.com", categories_sitemap)

    assert [candidate["listing_url"] for candidate in candidates] == [
        "https://example.com/kinh-te/",
        "https://example.com/chung-khoan/",
    ]


def test_extract_article_payload_strips_related_blocks_from_vneconomy_layout():
    html = """
    <html lang="vi">
      <head>
        <meta property="og:title" content="VCCI đề xuất giảm thuế" />
        <meta property="og:description" content="Sapo summary" />
      </head>
      <body>
        <main>
          <h1 class="name-detail">VCCI đề xuất giảm thuế</h1>
          <div class="main-detail-page">
            <div class="ct-edtior-web news-type1">
              <div class="news-sapo"><p>Sapo summary</p></div>
              <div>
                <p>Đoạn 1 của bài viết.</p>
                <p>Đoạn 2 của bài viết.</p>
              </div>
              <div class="list-detail-revert_item">
                <h3>Đọc thêm</h3>
                <p>Bài liên quan không nên xuất hiện.</p>
              </div>
              <div class="box-keyword">
                <h3>Từ khóa:</h3>
                <p>thuế bảo vệ môi trường</p>
              </div>
            </div>
          </div>
        </main>
      </body>
    </html>
    """

    payload = extract_article_payload("https://vneconomy.vn/example.htm", html)

    assert payload["title"] == "VCCI đề xuất giảm thuế"
    assert payload["excerpt"] == "Sapo summary"
    assert payload["content_text"] == "Sapo summary\nĐoạn 1 của bài viết.\nĐoạn 2 của bài viết."
    assert "Bài liên quan không nên xuất hiện." not in payload["content_text"]
    assert "Từ khóa:" not in payload["content_text"]


@pytest.mark.asyncio
async def test_discover_rss_feeds_follows_rss_hub_pages():
    homepage_html = """
    <html>
      <body>
        <a href="/rss.html">RSS</a>
      </body>
    </html>
    """
    hub_html = """
    <html>
      <body>
        <a href="/thi-truong/rss">Thi truong</a>
        <a href="/chung-khoan/rss">Chung khoan</a>
      </body>
    </html>
    """
    feed_xml = """
    <rss><channel>
      <item>
        <title>Headline</title>
        <link>https://example.com/articles/1</link>
        <description>Summary</description>
        <pubDate>Wed, 26 Mar 2026 10:30:00 +0700</pubDate>
      </item>
    </channel></rss>
    """

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text=homepage_html)
        if request.url.path == "/rss.html":
            return httpx.Response(200, text=hub_html)
        if request.url.path in {"/thi-truong/rss", "/chung-khoan/rss"}:
            return httpx.Response(200, text=feed_xml)
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://example.com") as client:
        discovered = await discover_rss_feeds(client, "https://example.com/")

    urls = {item["feed_url"] for item in discovered}
    assert "https://example.com/thi-truong/rss" in urls
    assert "https://example.com/chung-khoan/rss" in urls


@pytest.mark.asyncio
async def test_discover_rss_feeds_follows_plain_rss_directory_hub_pages():
    homepage_html = """
    <html>
      <body>
        <p>No explicit feed links on homepage.</p>
      </body>
    </html>
    """
    hub_html = """
    <html>
      <body>
        <a href="/rss/tin-moi-nhat.rss">Tin moi nhat</a>
        <a href="/rss/kinh-doanh.rss">Kinh doanh</a>
      </body>
    </html>
    """
    feed_xml = """
    <rss><channel>
      <item>
        <title>Headline</title>
        <link>https://example.com/articles/1</link>
        <description>Summary</description>
        <pubDate>Wed, 26 Mar 2026 10:30:00 +0700</pubDate>
      </item>
    </channel></rss>
    """

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text=homepage_html)
        if request.url.path == "/rss":
            return httpx.Response(200, text=hub_html)
        if request.url.path in {"/rss/tin-moi-nhat.rss", "/rss/kinh-doanh.rss"}:
            return httpx.Response(200, text=feed_xml)
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://example.com") as client:
        discovered = await discover_rss_feeds(client, "https://example.com/")

    urls = {item["feed_url"] for item in discovered}
    assert "https://example.com/rss/tin-moi-nhat.rss" in urls
    assert "https://example.com/rss/kinh-doanh.rss" in urls


@pytest.mark.asyncio
async def test_discover_crawl_listings_falls_back_to_category_sitemaps():
    homepage_html = """
    <html>
      <body>
        <p>No explicit feed links on homepage.</p>
      </body>
    </html>
    """
    sitemap_index = """
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap>
        <loc>https://example.com/sitemaps/categories.xml</loc>
        <lastmod>2026-03-27</lastmod>
      </sitemap>
    </sitemapindex>
    """
    categories_sitemap = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://example.com/kinh-te/</loc>
        <lastmod>2026-03-27T07:43:16+07:00</lastmod>
      </url>
      <url>
        <loc>https://example.com/chung-khoan/</loc>
        <lastmod>2026-03-27T07:43:16+07:00</lastmod>
      </url>
    </urlset>
    """

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text=homepage_html)
        if request.url.path in {"/rss", "/rss.xml", "/rss.html", "/rss.htm", "/feed", "/feed.xml", "/atom.xml"}:
            return httpx.Response(404, text="not found")
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, text=sitemap_index)
        if request.url.path == "/sitemaps/categories.xml":
            return httpx.Response(200, text=categories_sitemap)
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, base_url="https://example.com") as client:
        discovered = await discover_crawl_listings(client, "https://example.com/")

    assert [item["listing_url"] for item in discovered] == [
        "https://example.com/kinh-te/",
        "https://example.com/chung-khoan/",
    ]
    assert discovered[0]["discovery_method"] == "sitemap"


@pytest.mark.asyncio
async def test_store_article_candidate_dedupes_by_content_hash(db_session, monkeypatch):
    async def _fake_classify_article(title: str, excerpt: str | None, content_text: str | None):
        return {
            "topics": ["earnings"],
            "tickers": ["ABC"],
            "sectors": ["technology"],
            "importance": "high",
            "sentiment": "positive",
            "raw_payload": {"topics": ["earnings"]},
        }

    monkeypatch.setattr("app.services.news.service.classify_article", _fake_classify_article)

    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    first_feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    second_feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/markets.xml",
        title="Feed Two",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(first_feed)
    db_session.add(second_feed)
    await db_session.flush()

    stored_one = await news_service._store_article_candidate(
        db_session,
        source_type="rss",
        site_feed_id=first_feed.id,
        crawl_source_id=None,
        article_url="https://example.com/articles/first",
        title="ABC earnings rise",
        excerpt="Revenue improves",
        published_at=datetime(2026, 3, 26, 12, 0, 0),
        content_text="ABC earnings rise and revenue improves.",
        language="en",
        image_url=None,
    )
    stored_two = await news_service._store_article_candidate(
        db_session,
        source_type="rss",
        site_feed_id=second_feed.id,
        crawl_source_id=None,
        article_url="https://example.com/articles/duplicate",
        title="ABC earnings rise",
        excerpt="Revenue improves",
        published_at=datetime(2026, 3, 26, 12, 1, 0),
        content_text="ABC earnings rise and revenue improves.",
        language="en",
        image_url=None,
    )
    await db_session.commit()

    articles = (await db_session.execute(select(NewsArticle))).scalars().all()
    mappings = (await db_session.execute(select(NewsArticleSource))).scalars().all()
    semantics = (await db_session.execute(select(NewsArticleSemantic))).scalars().all()

    assert stored_one == 1
    assert stored_two == 1
    assert len(articles) == 1
    assert len(mappings) == 2
    assert len(semantics) == 1


@pytest.mark.asyncio
async def test_generate_article_summary_persists_and_prefers_llm_summary(db_session, monkeypatch):
    async def _fake_summarize_article(title: str, excerpt: str | None, content_text: str | None, *, language: str | None = None):
        assert title == "ABC earnings rise"
        assert excerpt == "Original excerpt"
        assert content_text == "Longer body text for the article."
        assert language == "en"
        return "LLM generated summary"

    monkeypatch.setattr("app.services.news.service.summarize_article", _fake_summarize_article)

    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://example.com/articles/first",
        title="ABC earnings rise",
        excerpt="Original excerpt",
        content_text="Longer body text for the article.",
        published_at=datetime(2026, 3, 26, 12, 0, 0),
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

    payload = await news_service.generate_article_summary(article.id, user_id=None)
    await db_session.refresh(article)

    assert payload is not None
    assert payload["excerpt"] == "LLM generated summary"
    assert payload["original_excerpt"] == "Original excerpt"
    assert payload["llm_summary"] == "LLM generated summary"
    assert article.llm_summary == "LLM generated summary"


@pytest.mark.asyncio
async def test_generate_article_summary_force_refresh_overwrites_existing_summary(db_session, monkeypatch):
    calls: list[str] = []

    async def _fake_summarize_article(title: str, excerpt: str | None, content_text: str | None, *, language: str | None = None):
        calls.append("called")
        return "Fresh regenerated summary"

    monkeypatch.setattr("app.services.news.service.summarize_article", _fake_summarize_article)

    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/rss.xml",
        title="Feed One",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://example.com/articles/force",
        title="ABC earnings rise",
        excerpt="Original excerpt",
        llm_summary="Old stored summary",
        content_text="Longer body text for the article.",
        published_at=datetime(2026, 3, 26, 12, 0, 0),
        language="en",
        content_hash="hash-force",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    await db_session.commit()

    payload = await news_service.generate_article_summary(article.id, user_id=None, force_refresh=True)
    await db_session.refresh(article)

    assert calls == ["called"]
    assert payload is not None
    assert payload["excerpt"] == "Fresh regenerated summary"
    assert payload["llm_summary"] == "Fresh regenerated summary"
    assert article.llm_summary == "Fresh regenerated summary"


@pytest.mark.asyncio
async def test_get_article_detail_repairs_stored_noisy_content(db_session, monkeypatch):
    html = """
    <html lang="vi">
      <head>
        <meta property="og:title" content="VCCI đề xuất giảm thuế" />
        <meta property="og:description" content="Sapo summary" />
      </head>
      <body>
        <main>
          <h1 class="name-detail">VCCI đề xuất giảm thuế</h1>
          <div class="main-detail-page">
            <div class="ct-edtior-web news-type1">
              <div class="news-sapo"><p>Sapo summary</p></div>
              <div>
                <p>Đoạn 1 của bài viết.</p>
                <p>Đoạn 2 của bài viết.</p>
              </div>
              <div class="list-detail-revert_item">
                <p>Bài liên quan không nên xuất hiện.</p>
              </div>
              <div class="box-keyword">
                <p>Từ khóa: thuế bảo vệ môi trường</p>
              </div>
            </div>
          </div>
        </main>
      </body>
    </html>
    """

    async def _fake_fetch_text(client, url: str) -> str:
        del client, url
        return html

    async def _fake_classify_article(title: str, excerpt: str | None, content_text: str | None):
        assert title == "VCCI đề xuất giảm thuế"
        assert excerpt == "Sapo summary"
        assert content_text == "Sapo summary\nĐoạn 1 của bài viết.\nĐoạn 2 của bài viết."
        return {
            "topics": ["tax"],
            "tickers": ["PLX"],
            "sectors": ["energy"],
            "importance": "high",
            "sentiment": "neutral",
            "raw_payload": {"topics": ["tax"]},
        }

    monkeypatch.setattr("app.services.news.service.fetch_text", _fake_fetch_text)
    monkeypatch.setattr("app.services.news.service.classify_article", _fake_classify_article)

    site = NewsSite(
        domain="vneconomy.vn",
        homepage_url="https://vneconomy.vn",
        display_name="VnEconomy",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://vneconomy.vn/tai-chinh/rss",
        title="Tài chính",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://vneconomy.vn/example.htm",
        title="VCCI đề xuất giảm thuế",
        excerpt="Old excerpt",
        content_text="Đảng Cộng sản Việt Nam - Đại hội XIV\nVới phương châm Đoàn kết - Dân chủ - Kỷ cương - Đột phá - Phát triển...",
        llm_summary="Old generated summary",
        published_at=datetime(2026, 3, 26, 18, 43, 0),
        language="vi",
        content_hash="noisy-hash",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    db_session.add(
        NewsArticleSemantic(
            article_id=article.id,
            topics=["politics"],
            tickers=[],
            sectors=["government"],
            importance="low",
            sentiment="neutral",
            raw_payload={},
        )
    )
    await db_session.commit()

    payload = await news_service.get_article_detail(article.id, user_id=None)
    await db_session.refresh(article)
    semantic = (await db_session.execute(select(NewsArticleSemantic).where(NewsArticleSemantic.article_id == article.id))).scalar_one()

    assert payload is not None
    assert payload["content_text"] == "Sapo summary\nĐoạn 1 của bài viết.\nĐoạn 2 của bài viết."
    assert "Bài liên quan" not in payload["content_text"]
    assert article.content_text == payload["content_text"]
    assert article.excerpt == "Sapo summary"
    assert article.llm_summary is None
    assert payload["topics"] == ["tax"]
    assert payload["tickers"] == ["PLX"]
    assert semantic.topics == ["tax"]
    assert semantic.sectors == ["energy"]


@pytest.mark.asyncio
async def test_refresh_article_content_returns_updated_detail(db_session, monkeypatch):
    html = """
    <html lang="vi">
      <head>
        <meta property="og:title" content="VCCI đề xuất giảm thuế" />
        <meta property="og:description" content="Sapo summary" />
      </head>
      <body>
        <main>
          <h1 class="name-detail">VCCI đề xuất giảm thuế</h1>
          <div class="main-detail-page">
            <div class="ct-edtior-web news-type1">
              <div class="news-sapo"><p>Sapo summary</p></div>
              <div>
                <p>Đoạn 1 của bài viết.</p>
                <p>Đoạn 2 của bài viết.</p>
              </div>
            </div>
          </div>
        </main>
      </body>
    </html>
    """

    async def _fake_fetch_text(client, url: str) -> str:
        del client, url
        return html

    async def _fake_classify_article(title: str, excerpt: str | None, content_text: str | None):
        assert excerpt == "Sapo summary"
        assert content_text == "Sapo summary\nĐoạn 1 của bài viết.\nĐoạn 2 của bài viết."
        return {
            "topics": ["tax"],
            "tickers": ["PLX"],
            "sectors": ["energy"],
            "importance": "high",
            "sentiment": "neutral",
            "raw_payload": {"topics": ["tax"]},
        }

    monkeypatch.setattr("app.services.news.service.fetch_text", _fake_fetch_text)
    monkeypatch.setattr("app.services.news.service.classify_article", _fake_classify_article)

    site = NewsSite(
        domain="vneconomy.vn",
        homepage_url="https://vneconomy.vn",
        display_name="VnEconomy",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://vneconomy.vn/tai-chinh.rss",
        title="Tài chính",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://vneconomy.vn/example-refresh.htm",
        title="VCCI đề xuất giảm thuế",
        excerpt="Old excerpt",
        content_text="Old content",
        llm_summary="Old generated summary",
        published_at=datetime(2026, 3, 26, 18, 43, 0),
        language="vi",
        content_hash="old-hash",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    db_session.add(
        NewsArticleSemantic(
            article_id=article.id,
            topics=["old-topic"],
            tickers=[],
            sectors=["old-sector"],
            importance="low",
            sentiment="neutral",
            raw_payload={},
        )
    )
    await db_session.commit()

    payload = await news_service.refresh_article_content(article.id, user_id=None)
    await db_session.refresh(article)
    semantic = (await db_session.execute(select(NewsArticleSemantic).where(NewsArticleSemantic.article_id == article.id))).scalar_one()

    assert payload is not None
    assert payload["content_text"] == "Sapo summary\nĐoạn 1 của bài viết.\nĐoạn 2 của bài viết."
    assert article.content_text == payload["content_text"]
    assert article.excerpt == "Sapo summary"
    assert article.llm_summary is None
    assert payload["topics"] == ["tax"]
    assert payload["tickers"] == ["PLX"]
    assert semantic.topics == ["tax"]
    assert semantic.sectors == ["energy"]


@pytest.mark.asyncio
async def test_get_article_detail_uses_site_domain_as_source_label(db_session):
    site = NewsSite(
        domain="cafef.vn",
        homepage_url="https://cafef.vn",
        display_name="CafeF",
        is_public=True,
    )
    db_session.add(site)
    await db_session.flush()

    feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://cafef.vn/rss.chn",
        title="A sample article title accidentally stored as feed title",
        kind="rss",
        discovery_method="manual",
        validation_status="valid",
        is_public=True,
    )
    db_session.add(feed)
    await db_session.flush()

    article = NewsArticle(
        canonical_url="https://cafef.vn/article-source-label.chn",
        title="Article title",
        excerpt="Excerpt",
        content_text="Body",
        published_at=datetime(2026, 3, 26, 12, 0, 0),
        language="vi",
        content_hash="source-label-hash",
    )
    db_session.add(article)
    await db_session.flush()
    db_session.add(NewsArticleSource(article_id=article.id, site_feed_id=feed.id, article_url=article.canonical_url))
    await db_session.commit()

    payload = await news_service.get_article_detail(article.id, user_id=None)

    assert payload is not None
    assert payload["source_labels"] == ["cafef.vn"]


def test_news_utc_naive_to_local_iso_uses_ho_chi_minh_timezone():
    assert utc_naive_to_local_iso(datetime(2026, 3, 26, 12, 14, 22)) == "2026-03-26T19:14:22+07:00"


def test_should_normalize_legacy_feed_title_flags_article_like_titles():
    site = NewsSite(domain="cafef.vn", homepage_url="https://cafef.vn", display_name="CafeF", is_public=True)

    assert news_service._should_normalize_legacy_feed_title(
        "Công ty chứng khoán sắp kinh doanh vàng phái sinh, lấn sân tài sản số",
        site,
    ) is True
    assert news_service._should_normalize_legacy_feed_title("CafeF RSS", site) is False
    assert news_service._should_normalize_legacy_feed_title("CafeF - Thị trường chứng khoán", site) is False


@pytest.mark.asyncio
async def test_normalize_legacy_rss_titles_updates_old_article_like_feed_titles(db_session):
    site = NewsSite(
        domain="cafef.vn",
        homepage_url="https://cafef.vn",
        display_name="CafeF",
        is_public=False,
    )
    db_session.add(site)
    await db_session.flush()

    bad_feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://cafef.vn/rss.chn",
        title="Công ty chứng khoán sắp kinh doanh vàng phái sinh, lấn sân tài sản số",
        kind="rss",
        discovery_method="manual",
        validation_status="valid",
        is_public=False,
    )
    good_feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://cafef.vn/thi-truong.rss",
        title="CafeF RSS",
        kind="rss",
        discovery_method="manual",
        validation_status="valid",
        is_public=False,
    )
    db_session.add_all([bad_feed, good_feed])
    await db_session.commit()

    changed_count = await news_service.normalize_legacy_rss_titles()
    await db_session.refresh(bad_feed)
    await db_session.refresh(good_feed)

    assert changed_count == 1
    assert bad_feed.title == "cafef.vn"
    assert good_feed.title == "CafeF RSS"


def test_news_admin_config_reads_and_writes_default_poll_interval(tmp_path, monkeypatch):
    settings_path = Path(tmp_path) / "settings.yaml"
    settings_path.write_text("brokers: []\n", encoding="utf-8")
    monkeypatch.setattr(news_service_module.settings, "settings_yaml_path", str(settings_path))
    monkeypatch.setattr(news_service_module.settings, "news_default_poll_interval_minutes", 30)

    assert news_service.get_default_poll_interval_minutes() == 30

    updated = news_service.update_admin_config(default_poll_interval_minutes=45)

    assert updated["default_poll_interval_minutes"] == 45
    assert news_service.get_default_poll_interval_minutes() == 45
    assert "default_poll_interval_minutes: 45" in settings_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_apply_default_poll_interval_to_existing_sources_updates_mismatched_sources(db_session, tmp_path, monkeypatch):
    settings_path = Path(tmp_path) / "settings.yaml"
    settings_path.write_text("news:\n  default_poll_interval_minutes: 30\n", encoding="utf-8")
    monkeypatch.setattr(news_service_module.settings, "settings_yaml_path", str(settings_path))

    site = NewsSite(
        domain="example.com",
        homepage_url="https://example.com",
        display_name="Example",
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
        poll_interval_minutes=60,
        next_poll_at=datetime(2026, 3, 26, 8, 0, 0),
    )
    source = NewsCrawlSource(
        site_id=site.id,
        listing_url="https://example.com/news",
        article_link_selector="article a",
        content_selector="article",
        excerpt_selector=None,
        pagination_config=None,
        validation_status="valid",
        is_public=False,
        poll_interval_minutes=45,
        next_poll_at=datetime(2026, 3, 26, 9, 0, 0),
    )
    unchanged_feed = NewsSiteFeed(
        site_id=site.id,
        feed_url="https://example.com/markets.xml",
        title="Markets RSS",
        kind="rss",
        discovery_method="seed",
        validation_status="valid",
        is_public=True,
        poll_interval_minutes=30,
        next_poll_at=datetime(2026, 3, 26, 10, 0, 0),
    )
    db_session.add_all([feed, source, unchanged_feed])
    await db_session.commit()

    original_utc_now = news_service_module.utc_now
    news_service_module.utc_now = lambda: datetime(2026, 3, 26, 12, 14, 22)
    try:
        changed_count = await news_service.apply_default_poll_interval_to_existing_sources()
    finally:
        news_service_module.utc_now = original_utc_now

    await db_session.refresh(feed)
    await db_session.refresh(source)
    await db_session.refresh(unchanged_feed)

    assert changed_count == 2
    assert feed.poll_interval_minutes == 30
    assert feed.next_poll_at == datetime(2026, 3, 26, 12, 14, 22)
    assert source.poll_interval_minutes == 30
    assert source.next_poll_at == datetime(2026, 3, 26, 12, 14, 22)
    assert unchanged_feed.poll_interval_minutes == 30
    assert unchanged_feed.next_poll_at == datetime(2026, 3, 26, 10, 0, 0)


@pytest.mark.asyncio
async def test_reconcile_stale_runs_marks_abandoned_running_rows_failed(db_session):
    stale_run = NewsIngestionRun(
        source_type="rss",
        status="running",
        started_at=datetime(2026, 3, 26, 12, 0, 0),
    )
    fresh_run = NewsIngestionRun(
        source_type="rss",
        status="running",
        started_at=datetime(2026, 3, 26, 12, NEWS_STALE_RUN_MINUTES, 1),
    )
    db_session.add(stale_run)
    db_session.add(fresh_run)
    await db_session.commit()

    original_utc_now = news_service_module.utc_now
    news_service_module.utc_now = lambda: datetime(2026, 3, 26, 12, NEWS_STALE_RUN_MINUTES, 1)
    try:
        cleaned = await news_service.reconcile_stale_runs()
    finally:
        news_service_module.utc_now = original_utc_now

    await db_session.refresh(stale_run)
    await db_session.refresh(fresh_run)

    assert cleaned == 1
    assert stale_run.status == "failed"
    assert stale_run.finished_at is not None
    assert fresh_run.status == "running"
    assert fresh_run.finished_at is None
